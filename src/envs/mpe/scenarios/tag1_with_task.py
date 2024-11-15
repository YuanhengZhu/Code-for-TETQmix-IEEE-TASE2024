import numpy as np
from ..core import World, Agent, Landmark
from ..scenario import BaseScenario
from ..utils import TaskGenerator
from ..utils import action_callback_tag1_adversary as action_callback


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        world.dim_c = 2

        task = TaskGenerator(type=3)
        # 用于确定最大的obs空间
        task.num_agents = 5
        task.num_adversaries = 4
        task.num_landmarks = 5
        self.reset_world(world, task)
        task_feats = len(task.code)
        self.entity_obs_feats = task_feats + 9   # pos(2) + vel(2) + is_self(1) + is_agent(1) + is_adversary(1) + is_landmark(1) + mask(1)
        self.entity_state_feats = task_feats + 8 # pos(2) + vel(2) + is_agent(1) + is_adversary(1) + is_landmark(1) + mask(1)
        return world

    def reset_world(self, world, task):
        bb = 1.2
        world.boundary = [np.array([bb, 0]), np.array([-bb, 0]), np.array([0, bb]), np.array([0, -bb])]
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
            if i < self.num_agents:
                agent.name = 'agent %d' % i
                agent.adversary = False
            else:
                agent.name = 'adversary %d' % (i - self.num_agents)
                agent.adversary = True
                agent.action_callback = action_callback
            # adversary被追捕，更小，更快
            # agent追捕，更大，更慢
            agent.size = 0.05 if agent.adversary else 0.075
            agent.accel = 4.0 if agent.adversary else 3.0
            agent.max_speed = 1.3 if agent.adversary else 1.0
        # add landmarks
        world.landmarks = [Landmark() for i in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        # set colors
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = world.np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = world.np_random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        if agent.name != 'agent 0':
            return 0
        
        rew = 0

        agents = [agent for agent in world.agents if not agent.adversary]
        adversaries = [agent for agent in world.agents if agent.adversary]
        for ag in agents:
            for adv in adversaries:
                if self.is_collision(ag, adv):
                    rew += 10

        rew = rew / self.num_agents

        return rew

    def entity_state(self, world):
        feats = np.zeros(self.entity_state_feats * self.num_entities)

        task_code = self.task.code
        
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
            feats[idx:idx+self.entity_state_feats] = task_code + [pos[0], pos[1], vel[0], vel[1], is_agent, is_adversary, is_landmark,] + [1.]
            idx += self.entity_state_feats

        # landmarks features
        for landmark in world.landmarks:
            is_agent = 0.
            is_adversary = 0.
            is_landmark = 1.
            pos = landmark.state.p_pos
            vel = landmark.state.p_vel
            feats[idx:idx+self.entity_state_feats] = task_code + [pos[0], pos[1], vel[0], vel[1], is_agent, is_adversary, is_landmark] + [1.]
            idx += self.entity_state_feats

        return feats

    def entity_observation(self, agent, world):
        feats = np.zeros(self.entity_obs_feats * self.num_entities)

        task_code = self.task.code
        
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
            feats[idx:idx+self.entity_obs_feats] = task_code + [pos[0], pos[1], vel[0], vel[1], is_self, is_agent, is_adversary, is_landmark] + [1.]
            idx += self.entity_obs_feats

        # landmarks features
        for landmark in world.landmarks:
            is_self = 0.
            is_agent = 0.
            is_adversary = 0.
            is_landmark = 1.
            pos = landmark.state.p_pos - pos_a
            vel = landmark.state.p_vel
            feats[idx:idx+self.entity_obs_feats] = task_code + [pos[0], pos[1], vel[0], vel[1], is_self, is_agent, is_adversary, is_landmark] + [1.]
            idx += self.entity_obs_feats

        return feats
