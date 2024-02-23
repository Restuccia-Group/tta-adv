import argparse
import os
import sys
import logging
import random
import torch
import numpy as np
from datetime import datetime
#from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode

_C = CfgNode()
cfg = _C
# ---------------- Model Options -------------#
_C.MODEL = CfgNode()
#Check https://github.com/RobustBench/robustbench for available models
_C.MODEL.ARCH = 'Standard'
_C.MODEL.ARCH2 = 'Standard'

#Choice of (source, norm, tent)
# - source : baseline without adaptation
# - norm : test-time normalzation
# - tent : test-time entropy minimization

_C.MODEL.ADAPTATION = 'source'

#to make adaptation episodic, reset the 
_C.MODEL.EPISODIC = False

_C.MODEL.CKPT_PATH = '.'
_C.MODEL.SAVE_PATH = '.'
_C.MODEL.EPS = 0.
_C.MODEL.LOSS = "polyloss"
_C.MODEL.DATASET = "cifar10"

# ----------------- Corruption Options --------------------#

_C.CORRUPTION = CfgNode()
_C.CORRUPTION.DATASET = 'cifar10'
_C.CORRUPTION.TYPE = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness', 'contrast',
                      'elastic_transform', 'pixelate', 'jpeg_compression'
                      ]
_C.CORRUPTION_SEVERITY = [3]
_C.CORRUPTION.NUM_EX = 10000

# ------------------ Batch Norm Options ------------------#

_C.BN = CfgNode()
_C.BN.EPSILON = 1e-5
_C.BN.MOMENTUM = 0.1

# ----------------- Optimizer Options -----------------#

_C.OPTIM = CfgNode()
_C.OPTIM.STEPS = 1
_C.OPTIM.LR = 1e-3
_C.OPTIM.METHOD = 'Adam'
_C.OPTIM.BETA = 0.9
_C.OPTIM.MOMENTUM = 0.9

_C.OPTIM.WD = 0.0
_C.OPTIM.TEMP = 1.0

_C.OPTIM.ADAPT = "ent"
_C.OPTIM.ADAPTIVE = False
_C.OPTIM.TBN = True
_C.OPTIM.UPDATE = True

# ----------------- Testing Option ----------------------#
_C.TEST = CfgNode()

_C.TEST.BATCH_SIZE = 128

_C.TEST.DATASET = "cifar10.1"

# -------------- Attacking Options ----------------------#

_C.ATTACK = CfgNode()
_C.ATTACK.METHOD = "PGD"
_C.ATTACK.SOURCE = 10
_C.ATTACK.EPS = 1.0
_C.ATTACK.ALPHA = 0.00392157
_C.ATTACK.STEPS = 500
_C.ATTACK.WHITE = True
_C.ATTACK.ADAPTIVE = False
_C.ATTACK.ADAPTIVE = False
_C.ATTACK.TARGETED = False
_C.ATTACK.PAR = 0.0
_C.ATTACK.WEIGHT_P = 0.0
_C.ATTACK.DEPRIOR = 0.0
_C.ATTACK.DFTESTPRIOR = 0.0
_C.ATTACK.LAYER = 0

# ---------------------- CUDNN Options -----------------------

_C.CUDNN = CfgNode()
_C.CUDNN.BENCHMARK = True

# ------------------------ Save and Load Config ---------------

_C.SAVE_DIR = "./output/test"
_C.DATA_DIR = "../../dataset"
_C.CKPT_DIR = "./ckpt"
_C.LOG_DEST = "log.txt"
_C.LOG_DIR = "./eval_results/tta"

# ---------------- Default Config ----------------------# 
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()

def assert_and_infer_cfg():
    err_str = "Unknown adaptation method."
    assert _C.MODEL.ADAPTATION in ["source","norm","tent"]
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)


def merge_from_file(cfg_file):
    with g_pathmgr.open(cfg_file, "r") as f:
        cfg = _C.load_cfg(f)
    _C.merge_from_other_cfg(cfg)


def dump_cfg():
    """
    Dumps Config file to the output directory
    """

    cfg_file = os.path.join(_C.SAVE_DIR, _C.CFG_DEST)
    with g_pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)

def load_cfg(out_dir, cfg_dest="config.yaml"):
    """
    Loads config from specified output directory"
    """
    cfg_file = os.path.join(out_dir, cfg_dest)
    merge_from_file(cfg_file)

def reset_cfg():
    """
    Reset configuration to default state
    """
    _C.merge_from_other_cfg(_CFG_DEFAULT)

def load_cfg_from_args():
    """
    Load config from command line arguments and set any specified options.
    """
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description= "config options")
    parser.add_argument("--cfg", dest="cfg_file",type=str, required=True,
                        help="config file location")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, 
                        help="See conf.py for all options")
