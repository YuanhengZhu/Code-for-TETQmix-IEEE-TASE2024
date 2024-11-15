from functools import partial

from .mpe.mpe_wrapper import MPEWrapper
# from .mamujoco.HalfCheetah_multitask import HalfCheetahMultiTask
from .multiagentenv import MultiAgentEnv


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["mpe"] = partial(env_fn, env=MPEWrapper)
# REGISTRY["mamujoco"] = partial(env_fn, env=HalfCheetahMultiTask)
