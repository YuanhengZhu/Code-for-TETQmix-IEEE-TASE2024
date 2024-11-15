REGISTRY = {}

from .basic_controller import BasicMAC
from .n_controller import NMAC
from .cqmix_controller import CQMixMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_mac"] = NMAC
REGISTRY["cqmix_mac"] = CQMixMAC
