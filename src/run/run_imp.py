import datetime
import os
import pprint
import threading
import time
from os.path import abspath, dirname
from types import SimpleNamespace as SN

import numpy as np
import torch as th

from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from utils.logging import Logger
from utils.timehelper import time_left, time_str


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    try:
        map_name = _config["env_args"]["map_name"]
    except:
        map_name = _config["env_args"]["key"]
    unique_token = (
        f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_seed{_config['seed']}"
    )

    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(dirname(abspath(__file__)))),
            args.local_results_path,
            "tb_logs",
            args.env,
            map_name,
            args.name,
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")


def evaluate_sequential(args, runner, test_episodes=10):
    save_dir = os.path.join(
        args.local_results_path,
        "renders",
        args.env,
        args.env_args["map_name"],
        args.name,
        f"{args.unique_token}",
    )
    if args.save_replay or args.save_animation:
        os.makedirs(save_dir, exist_ok=True)

    infos = []
    for i in range(test_episodes):
        episode_info = runner.run(
            test_mode=True,
            render=args.save_replay,
            save_animation=args.save_animation
            and (not args.evaluate or i % args.animation_interval_evaluation == 0),
            benchmark_mode=True,
        )
        for e_i in episode_info:
            e_i["episode"] = i + 1
        if args.save_replay:
            runner.save_replay(
                os.path.join(save_dir, f"render_episode_{runner.t_env}_{i}")
            )
        if args.save_animation and (
            not args.evaluate or i % args.animation_interval_evaluation == 0
        ):
            runner.save_animation(
                os.path.join(save_dir, f"animation_episode_{runner.t_env}_{i}")
            )
        infos.extend(episode_info)


def run_sequential(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    for k, v in env_info.items():
        setattr(args, k, v)

    logger.console_logger.info(f"Env info: {env_info}")

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    buffer0 = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )
    buffer1 = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )
    buffer2 = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )
    buffer3 = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )
    buffer4 = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )
    buffers = [buffer0, buffer1, buffer2, buffer3, buffer4]
    buffer_train = ReplayBuffer(
        scheme,
        groups,
        args.batch_size + 5,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer0.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer0.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
            )
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

    if args.evaluate or args.save_replay:
        runner.log_train_stats_t = runner.t_env
        evaluate_sequential(args, runner, test_episodes=args.test_nepisode)
        logger.log_stat("episode", runner.t_env, runner.t_env)
        logger.print_recent_stats()
        logger.console_logger.info("Finished Evaluation")
        return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_save_model_T = 0
    last_log_T = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    task_imp = [1]*5
    while runner.t_env <= args.t_max:
        # Run for a whole episode at a time
        prob_min = 0.1
        task_prob = np.array([prob_min]*5) + (1-prob_min*5) * np.exp(task_imp) / np.sum(np.exp(task_imp))
        task_type = np.random.choice(5, p=task_prob)
        if runner.t_env <= args.t_max * 0.1:
            task_prob = np.array([0.2]*5)
            task_type = (runner.t_env//25) % 5
        episode_batch = runner.run(test_mode=False, task_type=task_type)
        buffers[task_type].insert_episode_batch(episode_batch)

        # 上取整
        batch_size0 = int(args.batch_size * task_prob[0]) + 1
        batch_size1 = int(args.batch_size * task_prob[1]) + 1
        batch_size2 = int(args.batch_size * task_prob[2]) + 1
        batch_size3 = int(args.batch_size * task_prob[3]) + 1
        batch_size4 = int(args.batch_size * task_prob[4]) + 1
        batch_sizes = [batch_size0, batch_size1, batch_size2, batch_size3, batch_size4]

        for i, buffer in enumerate([buffer0, buffer1, buffer2, buffer3, buffer4]):
            if buffer.can_sample(batch_sizes[i]):
                episode_sample = buffer.sample(batch_sizes[i])
                buffer_train.insert_episode_batch(episode_sample)

        if buffer_train.can_sample(args.batch_size):
            episode_sample = buffer_train.sample(args.batch_size)      
            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]
            if episode_sample.device != args.device:
                episode_sample.to(args.device)
            learner.train(episode_sample, runner.t_env, episode)
        else:
            buffer_train.buffer_index = 0
            buffer_train.episodes_in_buffer = 0
        
        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()
            last_test_T = runner.t_env

            test_return_mean = []
            for task_type in range(5):
                test_returns = []
                for _ in range(n_test_runs):
                    test_return = runner.run(test_mode=True, task_type=task_type)
                    test_returns.append(test_return)
                test_return_mean.append(np.mean(test_returns))
            # 表现越差越重要
            task_imp = 1 - np.array(test_return_mean)
            task_imp = task_imp / 0.1

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

        if args.save_model and (runner.t_env - last_save_model_T) >= args.save_model_interval:
            save_path = os.path.join(args.local_results_path, "models", str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))
            learner.save_models(save_path)
            last_save_model_T = runner.t_env

    save_path = os.path.join(args.local_results_path, "models", str(runner.t_env))
    os.makedirs(save_path, exist_ok=True)
    logger.console_logger.info("Saving models to {}".format(save_path))
    learner.save_models(save_path)

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
