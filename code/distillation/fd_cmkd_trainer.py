# coding: utf-8
"""
fd_cmkd_trainer.py — Training manager for full FD-CMKD distillation.

Extends the Signformer TrainManager: at every training step the total loss is

    L = L_task(CTC)                                   (student recognition)
      + lambda_feat * (low_w * L_low + high_w * L_high)   (Eq. 8-9)
      + lambda_align * L_align                             (Eq. 10, shared CTC)

The teacher is frozen (precomputed features); the trainable extras are the
student-to-teacher projection and the two shared classifiers, all appended to
the student optimizer. Requires `import main.training` to resolve against
code/signformer (added to sys.path by the notebook).
"""
import itertools
import os
import shutil

import torch

import main.training as mt
from main.training import TrainManager
from fd_cmkd import (load_teacher_feats, build_teacher_batch,
                     build_shared_vocab, FDCMKDModule)


class FDCMKDTrainManager(TrainManager):
    """TrainManager with the full FD-CMKD objective added to each step."""

    def __init__(self, model, config):
        super().__init__(model=model, config=config)
        dc = config["training"].get("distillation", {}) or {}
        self.do_distill = bool(dc.get("enabled", False))
        if not self.do_distill:
            self.logger.info("FD-CMKD disabled — plain finetuning.")
            return

        # --- teacher features (frozen) ---
        self.teacher_feats = load_teacher_feats(
            os.path.join(dc["teacher_feats_dir"], "train.pkl"))
        if not self.teacher_feats:
            raise ValueError(
                f"No teacher features loaded from "
                f"{os.path.join(dc['teacher_feats_dir'], 'train.pkl')} "
                f"— run extract_skeleton_feats.py first and check the path."
            )
        teacher_dim = int(next(iter(self.teacher_feats.values())).shape[1])
        student_dim = config["model"]["encoder"]["hidden_size"]

        # --- shared vocabulary + lookup (student ids -> shared ids) ---
        self.shared_vocab, self.student_to_shared = build_shared_vocab(
            self.model.gls_vocab)
        if self.use_cuda:
            self.student_to_shared = self.student_to_shared.cuda()

        # --- trainable FD-CMKD components ---
        self.fd = FDCMKDModule(student_dim, teacher_dim, len(self.shared_vocab))
        if self.use_cuda:
            self.fd = self.fd.cuda()

        # --- loss weights ---
        self.lambda_feat = float(dc.get("lambda_feat", 0.5))
        self.low_w = float(dc.get("low_w", 1.0))
        self.high_w = float(dc.get("high_w", 0.25))
        self.lambda_align = float(dc.get("lambda_align", 0.3))
        self.align_ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)

        # append the new parameters to the existing optimizer, then keep the
        # scheduler's per-group lists consistent with the added group
        base_lr = self.optimizer.param_groups[0].get("lr", 1e-4)
        self.optimizer.add_param_group(
            {"params": list(self.fd.parameters()), "lr": base_lr})
        n_groups = len(self.optimizer.param_groups)
        for attr in ("min_lrs", "base_lrs"):
            lst = getattr(self.scheduler, attr, None)
            if isinstance(lst, list):
                while len(lst) < n_groups:
                    lst.append(lst[-1] if lst else 0.0)

        self.logger.info(
            "FD-CMKD ON | student_dim=%d teacher_dim=%d shared_vocab=%d | "
            "lambda_feat=%.3f low_w=%.2f high_w=%.2f lambda_align=%.3f | feats=%d",
            student_dim, teacher_dim, len(self.shared_vocab),
            self.lambda_feat, self.low_w, self.high_w, self.lambda_align,
            len(self.teacher_feats))

    def _train_batch(self, batch, update: bool = True):
        if not getattr(self, "do_distill", False):
            return super()._train_batch(batch, update=update)

        # --- student task loss (CTC over its own gloss head) ---
        encoder_output, _ = self.model.encode(
            sgn=batch.sgn, sgn_mask=batch.sgn_mask, sgn_length=batch.sgn_lengths)
        gloss_scores = self.model.gloss_output_layer(encoder_output)
        gloss_probs = gloss_scores.log_softmax(2).permute(1, 0, 2)
        task_loss = self.recognition_loss_function(
            gloss_probs, batch.gls,
            batch.sgn_lengths.long(), batch.gls_lengths.long()
        ) * self.recognition_loss_weight
        norm_task = task_loss / self.batch_multiplier

        # --- FD-CMKD terms ---
        t_batch, has_t = build_teacher_batch(
            encoder_output, batch.sgn_lengths, batch.sequence, self.teacher_feats)
        targets_shared = self.student_to_shared[batch.gls]
        fd = self.fd(encoder_output, t_batch, has_t, batch.sgn_lengths,
                     targets_shared, batch.gls_lengths, self.align_ctc,
                     low_w=self.low_w, high_w=self.high_w)

        total = norm_task + (self.lambda_feat * fd["feat"]
                             + self.lambda_align * fd["align"]) / self.batch_multiplier
        total.backward()

        if self.clip_grad_fun is not None:
            # self.fd is stepped by the optimizer (added as a param group above),
            # so it must be clipped too — otherwise the shared CTC classifiers
            # take unbounded updates while the student stays clipped.
            self.clip_grad_fun(
                params=itertools.chain(self.model.parameters(), self.fd.parameters()))
        if update:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.steps += 1
        if self.do_recognition:
            self.total_gls_tokens += batch.num_gls_tokens

        return norm_task, 0


def train_fd_cmkd(cfg_file: str) -> None:
    """Entry point mirroring main.training.train, using FDCMKDTrainManager."""
    cfg = mt.load_config(cfg_file)
    mt.set_seed(seed=cfg["training"].get("random_seed", 42))

    train_data, dev_data, test_data, gls_vocab, txt_vocab = mt.load_data(
        data_cfg=cfg["data"])

    do_recognition = cfg["training"].get("recognition_loss_weight", 1.0) > 0.0
    do_translation = cfg["training"].get("translation_loss_weight", 1.0) > 0.0
    multimodal = cfg["data"].get("multimodal", 1.0) > 0.0

    model = mt.build_model(
        cfg=cfg["model"], multimodal=multimodal,
        gls_vocab=gls_vocab, txt_vocab=txt_vocab,
        sgn_dim=sum(cfg["data"]["feature_size"])
        if isinstance(cfg["data"]["feature_size"], list)
        else cfg["data"]["feature_size"],
        do_recognition=do_recognition, do_translation=do_translation)

    trainer = FDCMKDTrainManager(model=model, config=cfg)
    os.makedirs(cfg["training"]["model_dir"], exist_ok=True)
    shutil.copy2(cfg_file, os.path.join(cfg["training"]["model_dir"], "config.yaml"))
    mt.log_data_info(train_data=train_data, valid_data=dev_data,
                     test_data=test_data, gls_vocab=gls_vocab,
                     txt_vocab=txt_vocab, logging_function=trainer.logger.info)
    gls_vocab.to_file(os.path.join(cfg["training"]["model_dir"], "gls.vocab"))
    txt_vocab.to_file(os.path.join(cfg["training"]["model_dir"], "txt.vocab"))

    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)
    del train_data, dev_data, test_data

    ckpt = "{}/{}.ckpt".format(trainer.model_dir, trainer.best_ckpt_iteration)
    output_path = os.path.join(
        trainer.model_dir, "best.IT_{:08d}".format(trainer.best_ckpt_iteration))
    logger = trainer.logger
    del trainer
    mt.test(cfg_file, ckpt=ckpt, output_path=output_path, logger=logger)
