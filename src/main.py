import collections
import os
import random
import sys
from copy import deepcopy
from os.path import abspath, dirname, join

import numpy as np
import torch as th
import yaml
from sacred import SETTINGS, Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from run import REGISTRY as run_REGISTRY
from utils.logging import get_logger

SETTINGS['CAPTURE_MODE'] = "fd" if os.name == 'nt' else "sys" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    

    config = config_copy(_config)

    # manual seed
    if config.get('manual_seed', False):
        config["seed"] = 42

    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    random.seed(config["seed"])
    config['env_args']['seed'] = config["seed"]
    
    # run
    run_REGISTRY[_config['run']](_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def my_parse_command(params, config_dict):
    for _i, _v in enumerate(params):
        key = _v.split("=")[0].strip()
        if key in config_dict:
            value = _v[_v.index('=')+1:].strip()
            if value == "True" or value == "true":
                value = True
            elif value == "False" or value == "false":
                value = False
            if isinstance(value, str):
                try:
                    value = int(value)
                except:
                    pass
            config_dict[key] = value
        elif key == "debug":
            config_dict['local_results_path'] = 'results/debug'
        elif key == "save_buffer":
            config_dict['run'] = 'run_save'


def main(params=None):
    if params is None:
        params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "env-config", "envs")
    alg_config = _get_config(params, "alg-config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # map name gives the name to the experiment
    if config_dict['env_args']['map_name']=='':
        if config_dict['env'] == 'mpe':
            config_dict['env_args']['map_name'] = f"{config_dict['env_args']['key']}"
        elif config_dict['env'] == 'mamujoco':
            config_dict['env_args']['map_name'] = f"{config_dict['env_args']['env_args']['scenario']}"
        else:
            raise NotImplementedError
    
    # Load the config from the command line
    my_parse_command(params, config_dict)

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    results_path = join(dirname(dirname(abspath(__file__))), config_dict["local_results_path"]) 
    file_obs_path = join(results_path, "sacred", config_dict['env'], config_dict['env_args']['map_name'], config_dict['name'])
    
    logger.info("Saving to FileStorageObserver in {}.".format(file_obs_path))
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params[0])

    # flush
    sys.stdout.flush()


if __name__ == '__main__':
    main()