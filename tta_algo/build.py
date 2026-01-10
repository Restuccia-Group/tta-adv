from tta_algo.rotta import RoTTA
from tta_algo.tent import TENT
from tta_algo.note import NOTE
from tta_algo.eata import EATA
from tta_algo.sar import SAR
from tta_algo.sotta import SoTTA
from tta_algo.cotta import CoTTA

def build_tta_adapter(cfg):

    if cfg.TTA.NAME == 'rotta':
        return RoTTA
    elif cfg.TTA.NAME == "tent":
        return TENT
    elif cfg.TTA.NAME == 'note':
        return NOTE
    elif cfg.TTA.NAME == 'eata':
        return EATA
    elif cfg.TTA.NAME == 'sar':
        return SAR
    elif cfg.TTA.NAME == 'sotta':
        return SoTTA
    elif cfg.TTA.NAME == 'cotta':
        return CoTTA
    else:
        raise NotImplementedError("The tta_adapter is not Implemented")
