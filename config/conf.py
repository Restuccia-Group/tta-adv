# import argparse
# import os
# import sys
# import logging
# import random
# import torch
import math
import numpy as np
from datetime import datetime
#from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode

_C = CfgNode()
cfg = _C

# ----------- Base Options ----------------- #
_C.BASE = CfgNode()
_C.BASE.SEED = 15
_C.BASE.NUM_WORKERS = 4
_C.BASE.GPU_ID = 0
_C.BASE.ATTACK = "dia"

# ---------------- Model Options -------------#
_C.MODEL = CfgNode()
#Check https://github.com/RobustBench/robustbench for available models
_C.MODEL.ARCH = 'SWin_b'
_C.MODEL.ADAPTATION = 'source'

#to make adaptation episodic, reset the 
_C.MODEL.EPISODIC = False

_C.MODEL.CKPT_PATH = '.'
_C.MODEL.SAVE_PATH = '.'
_C.MODEL.EPS = 0.

# --------------- DATASET Options ---------------------#
_C.DATA = CfgNode()
_C.DATA.PATH = "corrupted_data"
_C.DATA.SEVERITY = 3
_C.DATA.MODE = 'test'
_C.DATA.NUM_CLASSES = 10
_C.DATA.CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
_C.DATA.BATCH_SIZE = 200
_C.DATA.SIZE = (32,32)

# ----------------- Corruption Options --------------------#

_C.CORRUPTION = CfgNode()
_C.CORRUPTION.DATASET = 'cifar10c'
_C.CORRUPTION.TYPE = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness', 'contrast',
                      'elastic_transform', 'pixelate', 'jpeg_compression',
                      'gaussian_blur', 'shot_noise','saturate','spatter',
                      'normal', 'speckle_noise'
                      ]
_C.CORRUPTION.TEST = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness', 'contrast',
                      'elastic_transform', 'pixelate', 'jpeg_compression'
                      ]
_C.CORRUPTION.N_EXAMPLES = 10000

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

# _C.ATTACK = CfgNode()
# _C.ATTACK.METHOD = "PGD"
# _C.ATTACK.SOURCE = 10
# _C.ATTACK.EPS = 1.0
# _C.ATTACK.ALPHA = 0.00392157
# _C.ATTACK.STEPS = 500
# _C.ATTACK.WHITE = True
# _C.ATTACK.ADAPTIVE = False
# _C.ATTACK.ADAPTIVE = False
# _C.ATTACK.TARGETED = False
# _C.ATTACK.PAR = 0.0
# _C.ATTACK.WEIGHT_P = 0.0
# _C.ATTACK.DEPRIOR = 0.0
# _C.ATTACK.DFTESTPRIOR = 0.0
# _C.ATTACK.LAYER = 0

# ---------------------- CUDNN Options -----------------------

_C.CUDNN = CfgNode()
_C.CUDNN.BENCHMARK = True

# ------------------------ Save and Load Config ---------------

_C.SAVE_DIR = "./output/test"
_C.DATA_DIR = "corrupted_data"
_C.CKPT_DIR = "./ckpt"
_C.LOG_DEST = "log.txt"
_C.LOG_DIR = "./eval_results/tta"

# ------------------------- TTA Options --------------------
_C.TTA = CfgNode()

# RoTTA
_C.TTA.NAME = "tent"

_C.TTA.ROTTA = CfgNode()
_C.TTA.ROTTA.MEMORY_SIZE = 128
_C.TTA.ROTTA.UPDATE_FREQUENCY = 128
_C.TTA.ROTTA.NU = 0.001
_C.TTA.ROTTA.ALPHA = 0.05
_C.TTA.ROTTA.LAMBDA_T = 1.0
_C.TTA.ROTTA.LAMBDA_U = 1.0

# NOTE
_C.TTA.NOTE = CfgNode()
_C.TTA.NOTE.IABN_K = 3.0
_C.TTA.NOTE.TEMP = 1.0
_C.TTA.NOTE.MEMORY_SIZE = 64
_C.TTA.NOTE.UPDATE_FREQUENCY = 64
_C.TTA.NOTE.SKIP_THRESH = 0.2

# SAR
_C.TTA.SAR = CfgNode()
_C.TTA.SAR.STEPS = 1
_C.TTA.SAR.MARGIN_E0 = 0.4 * math.log(1000)
_C.TTA.SAR.EPISODIC = False
_C.TTA.SAR.RESET_CONST_EM = 0.2

# SOTTA
_C.TTA.SoTTA = CfgNode()
_C.TTA.SoTTA.MEM_SIZE = 64
_C.TTA.SoTTA.THRESH = 0.9



# -------------- Attacking Options ----------------------#

_C.DIA = CfgNode()
_C.DIA.METHOD = "PGD"
_C.DIA.MAL_PORTION = 0.2
_C.DIA.PSEUDO = False
_C.DIA.EPS = 8/255
_C.DIA.ALPHA = 2.0/255
_C.DIA.STEPS = 100
_C.DIA.WHITE = True
_C.DIA.ADAPTIVE = False
_C.DIA.TARGETED = False
_C.DIA.INIT = 'random'
_C.DIA.ADV_MODEL = False
_C.DIA.PAR = 0.0
_C.DIA.WEIGHT_P = 0.0
_C.DIA.DEPRIOR = 0.0
_C.DIA.DFTESTPRIOR = 0.0
_C.DIA.LAYER = 0
_C.DIA.PSEUDO = False
_C.DIA.CONTINUAL = False
_C.DIA.SRC_ONLY = False
_C.DIA.MED= False

