from rotta import RoTTA
from tent import TENT

def build_tta_adapter(cfg):

    if cfg.ADAPTER.NAME == 'rotta':
        return RoTTA
    elif cfg.ADAPTER.NAME == "tent":
        return TENT
    else:
        raise NotImplementedError("The tta_adapter is not Implemented")
