# coding: utf-8
"""
Collection of helper functions
"""

import copy
import glob
import os
import os.path
import errno
import re
import shutil
import random
import logging
from sys import platform
from logging import Logger
from typing import Callable, Optional
import numpy as np

import torch
from torch import nn, Tensor
from typing import Any as Dataset  # torchtext.data.Dataset replaced
import yaml
from main.vocabulary import GlossVocabulary, TextVocabulary


def make_model_dir(model_dir: str, overwrite: bool = False) -> str:
    """
    Create a new directory for the model.

    :param model_dir: path to model directory
    :param overwrite: whether to overwrite an existing directory
    :return: path to model directory
    """
    if os.path.isdir(model_dir):
        if not overwrite:
            raise FileExistsError("Model directory exists and overwriting is disabled.")
        # On Windows an open log file can't be deleted (WinError 32). Close any
        # logging handlers left open by a previous run before removing the dir.
        for h in logging.getLogger(__name__).handlers[:]:
            h.close()
            logging.getLogger(__name__).removeHandler(h)
        # delete previous directory to start with empty dir again
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    return model_dir


def make_logger(model_dir: str, log_file: str = "train.log") -> Logger:
    """
    Create a logger for logging the training process.

    :param model_dir: path to logging directory
    :param log_file: path to logging file
    :return: logger object
    """
    logger = logging.getLogger(__name__)
    # clear old handlers so the logger points to the current model_dir
    for h in logger.handlers[:]:
        h.close()
        logger.removeHandler(h)
    logger.setLevel(level=logging.DEBUG)
    fh = logging.FileHandler("{}/{}".format(model_dir, log_file), encoding='utf-8')
    fh.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.info("Hello! This is SL-CAT.")
    return logger


def log_cfg(cfg: dict, logger: Logger, prefix: str = "cfg"):
    """
    Write configuration to log.

    :param cfg: configuration to log
    :param logger: logger that defines where log is written to
    :param prefix: prefix for logging
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = ".".join([prefix, k])
            log_cfg(v, logger, prefix=p)
        else:
            p = ".".join([prefix, k])
            logger.info("{:34s} : {}".format(p, v))


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers. Transformer helper function.

    :param module: the module to clone
    :param n: clone this many times
    :return cloned modules
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype("uint8")
    return torch.from_numpy(mask) == 0


def set_seed(seed: int):
    """
    Set the random seed for modules torch, numpy and random.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_data_info(
    train_data: Dataset,
    valid_data: Dataset,
    test_data: Dataset,
    gls_vocab: GlossVocabulary,
    txt_vocab: TextVocabulary,
    logging_function: Callable[[str], None],
):
    """
    Log statistics of data and vocabulary.

    :param train_data:
    :param valid_data:
    :param test_data:
    :param gls_vocab:
    :param txt_vocab:
    :param logging_function:
    """
    logging_function(
        "Data set sizes: \n\ttrain {:d},\n\tvalid {:d},\n\ttest {:d}".format(
            len(train_data),
            len(valid_data),
            len(test_data) if test_data is not None else 0,
        )
    )

    logging_function(
        "First training example:\n\t[GLS] {}\n\t[TXT] {}".format(
            " ".join(vars(train_data[0])["gls"]), " ".join(vars(train_data[0])["txt"])
        )
    )

    logging_function(
        "First 10 words (gls): {}".format(
            " ".join("(%d) %s" % (i, t) for i, t in enumerate(gls_vocab.itos[:10]))
        )
    )
    logging_function(
        "First 10 words (txt): {}".format(
            " ".join("(%d) %s" % (i, t) for i, t in enumerate(txt_vocab.itos[:10]))
        )
    )

    logging_function("Number of unique glosses (types): {}".format(len(gls_vocab)))
    logging_function("Number of unique words (types): {}".format(len(txt_vocab)))


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def bpe_postprocess(string) -> str:
    """
    Post-processor for BPE output. Recombines BPE-split tokens.

    :param string:
    :return: post-processed string
    """
    return string.replace("@@ ", "")


def get_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory.
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir:
    :return: latest checkpoint file
    """
    list_of_files = glob.glob("{}/*.ckpt".format(ckpt_dir))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint


def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    # weights_only=False: i checkpoint contengono scalari numpy (np.core.multiarray.scalar).
    # Da PyTorch 2.6 il default e' True e fallirebbe con UnpicklingError. Sicuro: e' un nostro file.
    checkpoint = torch.load(
        path, map_location="cuda" if use_cuda else "cpu", weights_only=False
    )
    return checkpoint


# from onmt
def tile(x: Tensor, count: int, dim=0) -> Tensor:
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.

    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = (
        x.view(batch, -1)
        .transpose(0, 1)
        .repeat(count, 1)
        .transpose(0, 1)
        .contiguous()
        .view(*out_size)
    )
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def freeze_params(module: nn.Module):
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def symlink_update(target, link_name):
    """Aggiorna il puntatore 'best.ckpt'.

    Su Windows os.symlink richiede privilegi elevati (WinError 1314); in quel
    caso ripieghiamo su una copia fisica del checkpoint. target è relativo alla
    dir del link, quindi risolviamo il path assoluto per la copia.
    """
    # rimuovi eventuale link/file preesistente (gestisce EEXIST in modo uniforme)
    if os.path.islink(link_name) or os.path.exists(link_name):
        os.remove(link_name)
    try:
        os.symlink(target, link_name)
    except (OSError, NotImplementedError, AttributeError):
        # symlink non permesso (es. Windows senza developer mode) → copia fisica
        src = os.path.join(os.path.dirname(link_name), target)
        shutil.copyfile(src, link_name)


def parse_validations(model_dir: str):
    """Legge {model_dir}/validations.txt e ne estrae le curve.

    Ritorna un dict con liste parallele: steps, recognition_loss, wer, lr.
    I valori pari a -1 (metrica non calcolata) vengono lasciati come NaN così
    non sporcano i grafici.
    """
    path = os.path.join(model_dir, "validations.txt")
    steps, rloss, wer, lr = [], [], [], []
    if not os.path.exists(path):
        return {"steps": steps, "recognition_loss": rloss, "wer": wer, "lr": lr}

    def _num(pattern, line, default=float("nan")):
        m = re.search(pattern, line)
        if not m:
            return default
        v = float(m.group(1))
        return float("nan") if v == -1.0 else v

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip().startswith("Steps:"):
                continue
            s = re.search(r"Steps:\s*(\d+)", line)
            if not s:
                continue
            steps.append(int(s.group(1)))
            rloss.append(_num(r"Recognition Loss:\s*([-\d.]+)", line))
            wer.append(_num(r"WER\s+([-\d.]+)", line))
            lr.append(_num(r"LR:\s*([-\d.]+)", line))
    return {"steps": steps, "recognition_loss": rloss, "wer": wer, "lr": lr}


def plot_training_curves(model_dir: str, show: bool = False):
    """Plot recognition loss and dev WER curves.

    Reads {model_dir}/validations.txt and saves {model_dir}/training_curves.png.
    Returns the PNG path (or None if there is nothing to plot).
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")  # non-interactive backend for script saving
    import matplotlib.pyplot as plt

    data = parse_validations(model_dir)
    if not data["steps"]:
        return None

    steps = data["steps"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(steps, data["recognition_loss"], marker="o", ms=3, color="tab:blue")
    ax1.set_xlabel("step"); ax1.set_ylabel("recognition loss (dev)")
    ax1.set_title("Loss"); ax1.grid(alpha=0.3)

    wer = data["wer"]
    ax2.plot(steps, wer, marker="o", ms=3, color="crimson", label="dev WER")
    valid = [(s, w) for s, w in zip(steps, wer) if w == w]  # discard NaN
    if valid:
        bs, bw = min(valid, key=lambda t: t[1])
        ax2.scatter([bs], [bw], color="black", zorder=5, label=f"best {bw:.2f}%")
    ax2.set_xlabel("step"); ax2.set_ylabel("WER (%)")
    ax2.set_title("Dev WER"); ax2.grid(alpha=0.3); ax2.legend()

    fig.tight_layout()
    out = os.path.join(model_dir, "training_curves.png")
    fig.savefig(out, dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out
