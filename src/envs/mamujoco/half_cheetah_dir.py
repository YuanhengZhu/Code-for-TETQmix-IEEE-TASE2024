"""adapted from https:github.com/cnfinn/maml_rl/rllab/envs/mujoco"""
import numpy as np
from .multiagent_mujoco.mujoco_multi import MujocoMulti


class HalfCheetahDirEnvMulti(MujocoMulti):
    def __init__(
        self, n_agents, n_tasks=2, max_episode_steps=1000, task={"goal": 1.0}, **kwargs
    ):
        env_args = {
            "scenario": "HalfCheetah-v2",
            "agent_conf": "6x1",
            "agent_obsk": 1,
            "episode_limit": max_episode_steps,
        }
        self._task = task
        self._goal = task["goal"]
        self.num_tasks = n_tasks
        self.tasks = self.sample_tasks(n_tasks)
        self.max_episode_steps = max_episode_steps
        self.velocity = 0.0

        super(HalfCheetahDirEnvMulti, self).__init__(env_args=env_args)
        self.reset_task(0)

    def revise_reward(self, info):
        return self._task["direction"] * info["reward_run"] + info["reward_ctrl"]

    def step(self, actions):
        reward, done, info = super().step(actions)
        self.velocity = info["reward_run"]
        reward = self.revise_reward(info)
        info["goal"] = self.get_goal()
        obs_n = self.get_obs()
        reward_n = [np.array(reward) for _ in range(self.n_agents)]
        info["is_goal_state"] = True if info["reward_run"] * self._goal > 0 else False
        return obs_n, reward_n, done, info

    def sample_tasks(self, n_tasks):
        # -1后退, +1前进
        return [{"direction": -1.0}, {"direction": 1.0}]

    def set_task(self, task):
        self._task = task

    def get_task(self):
        return self._task

    def set_goal(self, goal):
        self._goal = goal
        self._task["direction"] = goal

    def get_goal(self):
        return self._goal

    def reset_task(self, task_idx=None):
        if task_idx is None:
            task_idx = np.random.randint(self.num_tasks)
        self._task = self.tasks[task_idx]
        self._goal = self._task["direction"]
        self.reset()
        self.velocity = 0.0

    def get_all_task_idx(self):
        return list(range(self.num_tasks))
