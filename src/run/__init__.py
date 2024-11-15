from .run import run as default_run
from .run_save import run as run_save
from .run_multi import run as run_multi
from .run_imp import run as run_imp
from .run_new import run as run_new
from .run_mamujoco import run as run_mamujoco


REGISTRY = {}
REGISTRY["default"] = default_run
REGISTRY["run_save"] = run_save
REGISTRY["run_multi"] = run_multi
REGISTRY["run_imp"] = run_imp
REGISTRY["run_new"] = run_new
REGISTRY["run_mamujoco"] = run_mamujoco
