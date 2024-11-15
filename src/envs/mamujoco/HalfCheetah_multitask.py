"""adapted from https:github.com/cnfinn/maml_rl/rllab/envs/mujoco"""
import numpy as np
from .multiagent_mujoco.mujoco_multi import MujocoMulti


class HalfCheetahMultiTask(MujocoMulti):
    def __init__(
        self,
        **kwargs
    ):       
        # 速度表（任务）
        self.velocitys = [2, 6, 10]
        self.set_task([1, 0, 1, 0, 0])
        env_args = kwargs["env_args"]
        super(HalfCheetahMultiTask, self).__init__(env_args=env_args)
        
        # 动作表
        # self.n_actions = 11
        # self.actions = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
        
        self.n_agents    = 6
        self.n_entities  = 6
        # self.obs_entity_feats = 4   # qpos, qvel, is_self, mask (qvel=0 if not self, mask=0 if not observed)
        # self.state_entity_feats = 3  # qpos, qvel, mask
        # self.obs_entity_feats = 10   # qpos, qvel, id(6), is_self, mask
        self.state_entity_feats = 9  # qpos, qvel, id(6), mask
        self.obs_entity_feats = 9   # qpos, qvel, id(6), mask
        self.task_feats = 5          # direction: 2, velocity: 3  
        
    def revise_reward(self, info):
        """
        根据step的info修正reward, 原来的reward越快越好, 现在的reward要保持速度
        gym.envs.mujoco.half_cheetah.HalfCheetahEnv.step()返回的info
        info = {"reward_run": (xposafter - xposbefore)/self.dt, 这个就是forward_vel
                "reward_ctrl": - 0.1 * np.square(action).sum()}

        reward_old = reward_run + reward_ctrl
        reward_new = reward_goal + reward_ctrl
        """
        forward_vel = info["reward_run"]
        reward_task = -1.0 * abs(forward_vel - self._goal) / abs(self._goal)
        # reward = 0.5 * info["reward_ctrl"] + reward_task
        return reward_task

    def step(self, actions):
        # actions = self.actions_map(actions)
        reward, done, info = super().step(actions)
        self.velocity = info["reward_run"]
        reward_m = self.revise_reward(info)
        # info["task"] = self._task
        # info["is_goal_state"] = (
        #     True if np.abs(info["reward_run"] - self._goal) <= 0.1 else False
        # )
        return reward_m, done, info

    def reset(self, task_type):
        super().reset()
        task = [0] * self.task_feats

        if task_type // 3 == 0:
            task[0] = 1
        elif task_type // 3 == 1:
            task[1] = 1

        if task_type % 3 == 0:
            task[2] = 1
        elif task_type % 3 == 1:
            task[3] = 1
        elif task_type % 3 == 2:
            task[4] = 1
            
        self.set_task(task)
    
    def get_obs(self):
        kdicts = self.k_dicts
        obs_n_original = super().get_obs()
        obs_n = []
        for i in range(self.n_agents):
            obs_i = []
            # for j in range(self.n_agents):
            #     agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
            #     agent_id_feats[j] = 1.0
            #     is_self = [1.0] if i == j else [0.0]
            #     mask = [1.0]
            #     agent_j_feats = np.concatenate(([obs_n_original[j][0], obs_n_original[j][1]], agent_id_feats, is_self, mask), axis=0)
            #     obs_i.append(agent_j_feats)

            # if len(kdicts[i][1]) == 1:
            #     obs_i.append([obs_n_original[i][0], obs_n_original[i][1], 1.0, 1.0])
            #     obs_i.append([obs_n_original[i][2], 0.0,                  0.0, 1.0])
            #     obs_i.append([0.0,                  0.0,                  0.0, 0.0])
            #     obs_i.append([0.0,                  0.0,                  0.0, 0.0])
            #     obs_i.append([0.0,                  0.0,                  0.0, 0.0])
            #     obs_i.append([0.0,                  0.0,                  0.0, 0.0])
            # elif len(kdicts[i][1]) == 2:
            #     obs_i.append([obs_n_original[i][0], obs_n_original[i][1], 1.0, 1.0])
            #     obs_i.append([obs_n_original[i][2], 0.0,                  0.0, 1.0])
            #     obs_i.append([obs_n_original[i][3], 0.0,                  0.0, 1.0])
            #     obs_i.append([0.0,                  0.0,                  0.0, 0.0])
            #     obs_i.append([0.0,                  0.0,                  0.0, 0.0])
            #     obs_i.append([0.0,                  0.0,                  0.0, 0.0])

            agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
            agent_id_feats[i] = 1.0
            mask = [1.0]
            agent_i_feats = np.concatenate(([obs_n_original[i][0], obs_n_original[i][1]], agent_id_feats, mask), axis=0)
            obs_i.append(agent_i_feats)
            for j in range(self.n_agents - 1):
                obs_i.append([0.0]*9)

            obs_i = np.concatenate(obs_i, axis=0)
            obs_n.append(obs_i)
        task_code = np.array(self._task).reshape(1, -1).repeat(self.n_agents, axis=0)
        obs_n = np.concatenate((obs_n, task_code), axis=1)

        return obs_n

    def get_state(self):
        obs_n_original = super().get_obs()
        state = []
        # for i in range(self.n_agents):
        #     state.append([obs_n_original[i][0], obs_n_original[i][1], 1.0])
        for i in range(self.n_agents):
            agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
            agent_id_feats[i] = 1.0
            mask = [1.0]
            agent_i_feats = np.concatenate(([obs_n_original[i][0], obs_n_original[i][1]], agent_id_feats, mask), axis=0)
            state.append(agent_i_feats)
        state = np.concatenate(state, axis=0)
        state = np.concatenate((state, self._task), axis=0)
        return state
    
    def get_obs_size(self):
        return len(self.get_obs()[0])

    def get_state_size(self):
        return len(self.get_state())

    def actions_map(self, actions):
        # 将离散的动作映射到连续的动作空间
        continuous_actions = []
        for action in actions:
            continuous_action = (self.actions[action])
            continuous_actions.append([continuous_action])
        return continuous_actions

    def set_task(self, task):
        self._task = task
        for i in range(2, 5):
            if task[i]:
                self._goal = self.velocitys[i - 2]
                break
        if task[1]:
            self._goal = -self._goal

    def get_task(self):
        return self._task
    
    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "obs_entity_feats": self.obs_entity_feats,
            "state_entity_feats": self.state_entity_feats,
            "n_actions": self.n_actions,
            "n_agents": self.n_agents,
            "n_entities": self.n_entities,
            "episode_limit": self.episode_limit,
            "task_feats": self.task_feats,
        }
        return env_info
