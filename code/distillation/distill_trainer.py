# coding: utf-8
import os
import shutil
import torch

# Import assoluti partendo dalla root del progetto (Signformer/)
# Assicurati che 'Signformer/' sia nel sys.path del tuo notebook
import main.training as mt
from main.training import TrainManager
from distillation.distill import load_teacher_feats, DistillHead, batch_distill_loss

# ... resto del tuo codice ...
class DistillTrainManager(TrainManager):
    """Estende il TrainManager di Signformer aggiungendo la loss di distillazione al train step."""

    def __init__(self, model, config):
        super().__init__(model=model, config=config)
        dc = config["training"].get("distillation", {}) or {}
        self.do_distill = bool(dc.get("enabled", False))
        if not self.do_distill:
            self.logger.info("Distillation DISABLED — training normale.")
            return

        feats_dir = dc["teacher_feats_dir"]
        self.teacher_feats = load_teacher_feats(os.path.join(feats_dir, "train.pkl"))
        if not self.teacher_feats:
            raise ValueError(
                f"No teacher features loaded from {os.path.join(feats_dir, 'train.pkl')} "
                f"— run extract_skeleton_feats.py first and check the path."
            )
        self.distill_lambda = float(dc.get("lambda", 1.0))
        self.distill_low_w  = float(dc.get("low_w", 1.0))
        self.distill_high_w = float(dc.get("high_w", 1.0))

        enc_dim = config["model"]["encoder"]["hidden_size"]
        teacher_dim = int(next(iter(self.teacher_feats.values())).shape[1])
        self.distill_proj = DistillHead(enc_dim, teacher_dim)
        if self.use_cuda:
            self.distill_proj = self.distill_proj.cuda()

        base_lr = self.optimizer.param_groups[0].get("lr", 1e-4)
        self.optimizer.add_param_group(
            {"params": list(self.distill_proj.parameters()), "lr": base_lr}
        )
        # >>> FIX SCHEDULER (vedi sopra) <<<
        n_groups = len(self.optimizer.param_groups)
        _min_lr = getattr(self, "learning_rate_min", 1e-8)
        for _attr in ("min_lrs", "base_lrs"):
            _lst = getattr(self.scheduler, _attr, None)
            if isinstance(_lst, list):
                while len(_lst) < n_groups:
                    _lst.append(_lst[-1] if _lst else _min_lr)

        self.logger.info(
            "Distillation ON | student_dim=%d teacher_dim=%d | lambda=%.3f low_w=%.2f high_w=%.2f | N feats=%d",
            enc_dim, teacher_dim, self.distill_lambda, self.distill_low_w,
            self.distill_high_w, len(self.teacher_feats),
        )

    def _train_batch(self, batch, update: bool = True):
        # senza distillazione: comportamento identico all'originale
        if not getattr(self, "do_distill", False):
            return super()._train_batch(batch, update=update)

        # --- recognition (CTC) ricalcolata per riusare encoder_output ---
        encoder_output, _ = self.model.encode(
            sgn=batch.sgn, sgn_mask=batch.sgn_mask, sgn_length=batch.sgn_lengths
        )
        gloss_scores = self.model.gloss_output_layer(encoder_output)
        gloss_probs = gloss_scores.log_softmax(2).permute(1, 0, 2)     # (T, N, C)
        recognition_loss = self.recognition_loss_function(
            gloss_probs, batch.gls, batch.sgn_lengths.long(), batch.gls_lengths.long()
        ) * self.recognition_loss_weight
        norm_rec = recognition_loss / self.batch_multiplier

        # --- distillazione cross-modale FD-CMKD ---
        distill = batch_distill_loss(
            encoder_output, batch.sgn_lengths, batch.sequence,
            self.teacher_feats, self.distill_proj,
            self.distill_low_w, self.distill_high_w,
        )
        total_loss = norm_rec + self.distill_lambda * distill / self.batch_multiplier
        total_loss.backward()

        if self.clip_grad_fun is not None:
            self.clip_grad_fun(params=self.model.parameters())
        if update:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.steps += 1
        if self.do_recognition:
            self.total_gls_tokens += batch.num_gls_tokens

        return norm_rec, 0


def train_distill(cfg_file: str) -> None:
    """Come main.training.train ma usa DistillTrainManager (finetuning con distillazione)."""
    cfg = mt.load_config(cfg_file)
    mt.set_seed(seed=cfg["training"].get("random_seed", 42))

    train_data, dev_data, test_data, gls_vocab, txt_vocab = mt.load_data(data_cfg=cfg["data"])

    do_recognition = cfg["training"].get("recognition_loss_weight", 1.0) > 0.0
    do_translation = cfg["training"].get("translation_loss_weight", 1.0) > 0.0
    multimodal = cfg["data"].get("multimodal", 1.0) > 0.0

    model = mt.build_model(
        cfg=cfg["model"],
        multimodal=multimodal,
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        sgn_dim=sum(cfg["data"]["feature_size"])
        if isinstance(cfg["data"]["feature_size"], list)
        else cfg["data"]["feature_size"],
        do_recognition=do_recognition,
        do_translation=do_translation,
    )

    trainer = DistillTrainManager(model=model, config=cfg)
    shutil.copy2(cfg_file, trainer.model_dir + "/config.yaml")
    mt.log_cfg(cfg, trainer.logger)
    mt.log_data_info(
        train_data=train_data, valid_data=dev_data, test_data=test_data,
        gls_vocab=gls_vocab, txt_vocab=txt_vocab, logging_function=trainer.logger.info,
    )
    gls_vocab.to_file("{}/gls.vocab".format(cfg["training"]["model_dir"]))
    txt_vocab.to_file("{}/txt.vocab".format(cfg["training"]["model_dir"]))

    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)
    del train_data, dev_data, test_data

    try:
        png = mt.plot_training_curves(trainer.model_dir)
        if png:
            trainer.logger.info("Grafici di training salvati in: %s", png)
    except Exception as e:
        trainer.logger.warning("Impossibile generare i grafici: %s", e)

    ckpt = "{}/{}.ckpt".format(trainer.model_dir, trainer.best_ckpt_iteration)
    output_path = os.path.join(
        trainer.model_dir, "best.IT_{:08d}".format(trainer.best_ckpt_iteration)
    )
    logger = trainer.logger
    del trainer
    mt.test(cfg_file, ckpt=ckpt, output_path=output_path, logger=logger)
