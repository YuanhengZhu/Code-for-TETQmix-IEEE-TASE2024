import numpy as np
from ..core import World, Agent, Landmark
from ..scenario import BaseScenario
from ..utils import TaskGenerator
from ..utils import action_callback_push_adversary as action_callback

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        world.dim_c = 2

        task = TaskGenerator(type=2)
        # 用于确定最大的obs空间
        task.num_agents = 5
        task.num_adversaries = 4
        task.num_landmarks = 5
        self.reset_world(world, task)
        self.entity_obs_feats = 9   # pos(2) + vel(2) + is_self(1) + is_agent(1) + is_adversary(1) + is_landmark(1) + mask(1)
        self.entity_state_feats = 8 # pos(2) + vel(2) + is_agent(1) + is_adversary(1) + is_landmark(1) + mask(1)
        return world

    def reset_world(self, world, task):
        self.task = task
        self.num_agents = task.num_agents
        self.num_adversaries = task.num_adversaries
        self.num_landmarks = task.num_landmarks
        self.num_entities = self.num_agents + self.num_adversaries + self.num_landmarks
        # add agents
        world.agents = [Agent() for i in range(self.num_agents + self.num_adversaries)]
        for i, agent in enumerate(world.agents):
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            if i < self.num_agents:
                agent.name = 'agent %d' % i
                agent.adversary = False
            else:
                agent.name = 'adversary %d' % (i - self.num_agents)
                agent.adversary = True
                agent.action_callback = action_callback
        # add landmarks
        world.landmarks = [Landmark() for i in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.08
        # set color for agents
        for i, agent in enumerate(world.agents):
            if agent.adversary:
                agent.color = np.array([0.85, 0.35, 0.35])
            else:
                agent.color = np.array([0.35, 0.85, 0.35])
        # set color for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            t = 0
            while True:
                landmark.state.p_pos = np.random.uniform(-1-0.1*t, +1+0.1*t, world.dim_p)
                if i == 0:
                    break
                if min([np.linalg.norm(landmark.state.p_pos - world.landmarks[j].state.p_pos) for j in range(i)]) > 0.5:
                    break
                t += 1
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        if agent.name != 'agent 0':
            return 0
        
        rew = 0

        for landmark in world.landmarks:
            agent_dist = [np.sqrt(np.sum(np.square(landmark.state.p_pos - a.state.p_pos))) for a in world.agents if not a.adversary]
            adv_dist = [np.sqrt(np.sum(np.square(landmark.state.p_pos - a.state.p_pos))) for a in world.agents if a.adversary]
            rew -= min(agent_dist) # agent should go to the landmark
            rew += min(adv_dist)   # agent should keep adversary away from the landmark
        
        rew = rew / self.num_agents

        return rew
               
    def entity_state(self, world):
        feats = np.zeros(self.entity_state_feats * self.num_entities)
        
        # agents features
        idx = 0
        for a in world.agents:
            pos = a.state.p_pos
            vel = a.state.p_vel
            if a.adversary:
                is_agent = 0.
                is_adversary = 1.
                is_landmark = 0.
            else:
                is_agent = 1.
                is_adversary = 0.
                is_landmark = 0.
            feats[idx:idx+self.entity_state_feats] = [pos[0], pos[1], vel[0], vel[1], is_agent, is_adversary, is_landmark,] + [1.]
            idx += self.entity_state_feats

        # landmarks features
        for landmark in world.landmarks:
            is_agent = 0.
            is_adversary = 0.
            is_landmark = 1.
            pos = landmark.state.p_pos
            vel = landmark.state.p_vel
            feats[idx:idx+self.entity_state_feats] = [pos[0], pos[1], vel[0], vel[1], is_agent, is_adversary, is_landmark] + [1.]
            idx += self.entity_state_feats

        return feats

    def entity_observation(self, agent, world):
        feats = np.zeros(self.entity_obs_feats * self.num_entities)
        
        # agent features
        pos_a = agent.state.p_pos
        idx = 0
        for a in world.agents:
            if a is agent:
                is_self = 1.
            else:
                is_self = 0.
            if a.adversary:
                is_agent = 0.
                is_adversary = 1.
                is_landmark = 0.
            else:
                is_agent = 1.
                is_adversary = 0.
                is_landmark = 0.
            pos = a.state.p_pos - pos_a
            vel = a.state.p_vel
            feats[idx:idx+self.entity_obs_feats] = [pos[0], pos[1], vel[0], vel[1], is_self, is_agent, is_adversary, is_landmark] + [1.]
            idx += self.entity_obs_feats

        # landmarks features
        for landmark in world.landmarks:
            is_self = 0.
            is_agent = 0.
            is_adversary = 0.
            is_landmark = 1.
            pos = landmark.state.p_pos - pos_a
            vel = landmark.state.p_vel
            feats[idx:idx+self.entity_obs_feats] = [pos[0], pos[1], vel[0], vel[1], is_self, is_agent, is_adversary, is_landmark] + [1.]
            idx += self.entity_obs_feats

        return feats
