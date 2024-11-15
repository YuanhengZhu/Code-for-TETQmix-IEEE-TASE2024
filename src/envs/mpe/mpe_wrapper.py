import importlib

import numpy as np
from gym.spaces import flatdim
from gym.wrappers import TimeLimit as GymTimeLimit

from .environment import MultiAgentEnv


class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not all(done)
            done = len(observation) * [True]
        return observation, reward, done, info


class MPEWrapper(MultiAgentEnv):
    def __init__(self, key, time_limit, **kwargs):

        self.episode_limit = time_limit

        scenario = importlib.import_module("envs.mpe.scenarios."+key).Scenario()
        world = scenario.make_world()

        if 'spread' in key: type = 0
        elif 'form_shape' in key: type = 1
        elif 'push' in key:
            if 'away' in key:
                type = 5
            else:
                type = 2
        elif 'tag1' in key: type = 3
        elif 'tag2' in key: type = 4
        elif 'multi' in key: type = None
        env = MultiAgentEnv(
            world,
            reset_callback=scenario.reset_world,
            reward_callback=scenario.reward,
            observation_callback=scenario.entity_observation,
            state_callback=scenario.entity_state,
            world_info_callback=getattr(scenario, "world_benchmark_data", None),
            type=type
        )

        self._env = TimeLimit(env, max_episode_steps=time_limit)

        # basic env variables
        self.n_agents    = self._env.n_agents
        self.n_adversaries = getattr(scenario, "num_adversaries", 0)
        self.n_landmarks = getattr(scenario, "num_landmarks", len(self._env.world.landmarks)) 
        self.n_entities  = getattr(scenario, "num_entities", self.n_agents+self.n_landmarks)
        self.n_entity_types = getattr(scenario, "num_entity_types", 2)
        self._obs = None
        self._state = None
        self._task = None

        self.obs_entity_feats = getattr(scenario, "entity_obs_feats", 0)
        self.state_entity_feats = getattr(scenario, "entity_state_feats", 0)
        self.task_feats = len(scenario.task.code)

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        # check if the scenario uses an entity state 
        self.custom_state = kwargs.get("state_entity_mode", False)
        self.custom_state_dim = self.state_entity_feats * self.n_entities + self.task_feats

        self._seed = kwargs["seed"]
        self._env.seed(self._seed)

    def step(self, actions, animate=False):
        """ Returns reward, terminated, info """
        actions = [int(a) for a in actions]
        self._obs, reward, done, info = self._env.step(actions)
        self._obs = np.array(self._obs)
        num_agents = self._task.num_agents
        num_adversaries = self._task.num_adversaries
        num_landmarks = self._task.num_landmarks
        idx1 = num_agents * self.obs_entity_feats
        idx2 = idx1 + num_adversaries * self.obs_entity_feats
        idx3 = idx2 + num_landmarks * self.obs_entity_feats
        self._obs = np.concatenate((self._obs[:, :idx1], 
                                    np.zeros((len(self._obs), (self.n_agents-num_agents)*self.obs_entity_feats)), 
                                    self._obs[:, idx1:idx2], 
                                    np.zeros((len(self._obs), (self.n_adversaries-num_adversaries)*self.obs_entity_feats)),
                                    self._obs[:, idx2:],
                                    np.zeros((len(self._obs), (self.n_landmarks-num_landmarks)*self.obs_entity_feats))),
                                    axis=1)
        self._obs = np.concatenate((self._obs, np.zeros((self.n_agents-len(self._obs), flatdim(self.longest_observation_space)))), axis=0)
        task_code = np.array(self._task.code).reshape(1, -1).repeat(self.n_agents, axis=0)
        self._obs = np.concatenate((self._obs, task_code), axis=1)

        if self.custom_state:
            self._state = self._env.get_state()
            self._state = np.array(self._state)
            idx1 = num_agents * self.state_entity_feats
            idx2 = idx1 + num_adversaries * self.state_entity_feats
            idx3 = idx2 + num_landmarks * self.state_entity_feats
            self._state = np.concatenate((self._state[:idx1], 
                                          np.zeros((self.n_agents-num_agents)*self.state_entity_feats),
                                          self._state[idx1:idx2],
                                          np.zeros((self.n_adversaries-num_adversaries)*self.state_entity_feats),
                                          self._state[idx2:],
                                          np.zeros((self.n_landmarks-num_landmarks)*self.state_entity_feats)), 
                                          axis=0)
            self._state = np.concatenate((self._state, self._task.code))

        return float(sum(reward)), all(done), info["world"]

    def get_agent_positions(self):
        """Returns the [x,y] positions of each agent in a list"""
        return [a.state.p_pos for a in self._env.world.agents]

    def get_landmark_positions(self):
        return [l.state.p_pos for l in self._env.world.landmarks]

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return flatdim(self.longest_observation_space) + self.task_feats

    def get_state(self):
        if self.custom_state:
            return self._state
        else: 
            return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        if self.custom_state:
            return self.custom_state_dim
        else:
            return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def reset(self, task_type=None):
        """ Returns initial observations and states"""
        self._obs = self._env.reset(task_type=task_type)
        self._task = self._env.task
        num_agents = self._task.num_agents
        num_adversaries = self._task.num_adversaries
        num_landmarks = self._task.num_landmarks
        self._obs = np.array(self._obs)
        idx1 = num_agents * self.obs_entity_feats
        idx2 = idx1 + num_adversaries * self.obs_entity_feats
        idx3 = idx2 + num_landmarks * self.obs_entity_feats
        self._obs = np.concatenate((self._obs[:, :idx1], 
                                    np.zeros((len(self._obs), (self.n_agents-num_agents)*self.obs_entity_feats)), 
                                    self._obs[:, idx1:idx2], 
                                    np.zeros((len(self._obs), (self.n_adversaries-num_adversaries)*self.obs_entity_feats)),
                                    self._obs[:, idx2:],
                                    np.zeros((len(self._obs), (self.n_landmarks-num_landmarks)*self.obs_entity_feats))),
                                    axis=1)
        self._obs = np.concatenate((self._obs, np.zeros((self.n_agents-len(self._obs), flatdim(self.longest_observation_space)))), axis=0)
        task_code = np.array(self._task.code).reshape(1, -1).repeat(self.n_agents, axis=0)
        self._obs = np.concatenate((self._obs, task_code), axis=1)

        if self.custom_state:
            self._state = self._env.get_state()
            self._state = np.array(self._state)
            idx1 = num_agents * self.state_entity_feats
            idx2 = idx1 + num_adversaries * self.state_entity_feats
            idx3 = idx2 + num_landmarks * self.state_entity_feats
            self._state = np.concatenate((self._state[:idx1], 
                                          np.zeros((self.n_agents-num_agents)*self.state_entity_feats),
                                          self._state[idx1:idx2],
                                          np.zeros((self.n_adversaries-num_adversaries)*self.state_entity_feats),
                                          self._state[idx2:],
                                          np.zeros((self.n_landmarks-num_landmarks)*self.state_entity_feats)), 
                                          axis=0)
            self._state = np.concatenate((self._state, self._task.code))

        return self.get_obs(), self.get_state()

    def render(self):
        return self._env.render(mode='rgb_array')

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "obs_entity_feats": self.obs_entity_feats,
            "state_entity_feats": self.state_entity_feats,
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "n_entities": self.n_entities,
            "episode_limit": self.episode_limit,
            "task_feats": self.task_feats,
        }
        return env_info
    
    def get_task(self):
        return self._task