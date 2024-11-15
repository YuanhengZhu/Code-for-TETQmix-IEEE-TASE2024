import numpy as np

from ..core import Agent, Landmark, World
from ..scenario import BaseScenario
from ..utils import TaskGenerator
from ..utils import action_callback_push_adversary as push_action_callback
from ..utils import action_callback_tag1_adversary as tag1_action_callback
from ..utils import action_callback_tag2_adversary as tag2_action_callback


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
        self.entity_obs_feats = 9   # pos(2) + vel(2) + is_self(1) + is_agent(1) + is_adversary(1) + is_landmark(1) + mask(1)
        self.entity_state_feats = 8 # pos(2) + vel(2) + is_agent(1) + is_adversary(1) + is_landmark(1) + mask(1)
        return world

    def reset_world(self, world, task):
        self.task = task
        if task.type == 3 or task.type == 4:
            bb = 1.2
            world.boundary = [np.array([bb, 0]), np.array([-bb, 0]), np.array([0, bb]), np.array([0, -bb])]
        else:
            world.boundary = None
        self.num_agents = task.num_agents
        self.num_adversaries = task.num_adversaries
        self.num_landmarks = task.num_landmarks
        self.num_entities = self.num_agents + self.num_adversaries + self.num_landmarks
        # add agents
        world.agents = [Agent() for i in range(self.num_agents + self.num_adversaries)]
        for i, agent in enumerate(world.agents):
            agent.movable = True
            agent.collide = True
            agent.silent = True
            if i < self.num_agents:
                agent.name = 'agent %d' % i
                agent.adversary = False
            else:
                agent.name = 'adversary %d' % (i - self.num_agents)
                agent.adversary = True
                if task.type == 2:
                    agent.action_callback = push_action_callback
                elif task.type == 3:
                    agent.action_callback = tag1_action_callback
                elif task.type == 4:
                    agent.action_callback = tag2_action_callback
            if task.type == 3:
                agent.size = 0.075 if not agent.adversary else 0.05
                agent.accel = 3.0 if not agent.adversary else 4.0
                agent.max_speed = 1.0 if not agent.adversary else 1.3
            elif task.type == 4:
                agent.size = 0.05 if not agent.adversary else 0.075
                agent.accel = 4.0 if not agent.adversary else 3.0
                agent.max_speed = 1.3 if not agent.adversary else 1.0
            else:
                agent.size = 0.15

        # add landmarks
        world.landmarks = [Landmark() for i in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            if task.type == 3 or task.type == 4:
                landmark.size = 0.2
                landmark.collide = True
            else:
                landmark.size = 0.08
                landmark.collide = False
            landmark.movable = False
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
        if task.type == 3 or task.type == 4:
            for i, landmark in enumerate(world.landmarks):
                if not landmark.boundary:
                    landmark.state.p_pos = world.np_random.uniform(-0.9, +0.9, world.dim_p)
                    landmark.state.p_vel = np.zeros(world.dim_p)
        else:
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

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        if agent.name != "agent 0":
            return 0

        rew = 0

        if self.task.type == 0: # spread
            for landmark in world.landmarks:
                dist = [np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos)))
                        for a in world.agents]
                rew -= min(dist)
            for agent in world.agents:
                if agent.collide:
                    for a in world.agents:
                        if a is not agent and self.is_collision(a, agent):
                            rew -= 1
            rew = rew / self.num_agents
            r1 = -12
            r2 = -16.36
            rew = (rew - r2/25) / (r1 - r2)
        elif self.task.type == 1:
            targets = []
            shape = self.task.shape
            center = world.landmarks[0].state.p_pos
            # 1 正多边形 2 ─ 3 └
            d = 1
            if shape == 1:
                for i in range(self.num_agents):
                    delta = d * np.array([np.cos(2 * np.pi * i / self.num_agents), np.sin(2 * np.pi * i / self.num_agents)])
                    targets.append(center + delta)
            elif shape == 2:
                for i in range(self.num_agents):
                    delta = d * (-1)**i * ((i+1)//2) * np.array([1, 0])
                    targets.append(center + delta)
            elif shape == 3:
                for i in range(self.num_agents):
                    if i % 2 == 0:
                        delta = d * ((i+1)//2) * np.array([1, 0])
                    else:
                        delta = d * ((i+1)//2) * np.array([0, 1])
                    targets.append(center + delta)
            for t in targets:
                dist = [np.sqrt(np.sum(np.square(a.state.p_pos - t)))
                        for a in world.agents]
                rew -= min(dist)
            for agent in world.agents:
                if agent.collide:
                    for a in world.agents:
                        if a is not agent and self.is_collision(a, agent):
                            rew -= 1
            rew = rew / self.num_agents
            r1 = -15.74
            r2 = -21.85
            rew = (rew - r2/25) / (r1 - r2)
        elif self.task.type == 2:
            for landmark in world.landmarks:
                agent_dist = [np.sqrt(np.sum(np.square(landmark.state.p_pos - a.state.p_pos))) for a in world.agents if not a.adversary]
                adv_dist = [np.sqrt(np.sum(np.square(landmark.state.p_pos - a.state.p_pos))) for a in world.agents if a.adversary]
                rew -= min(agent_dist) # agent should go to the landmark
                rew += min(adv_dist)   # agent should keep adversary away from the landmark
            rew = rew / self.num_agents
            r1 = -4.58
            r2 = -11.51
            rew = (rew - r2/25) / (r1 - r2)
        elif self.task.type == 3:
            agents = [agent for agent in world.agents if not agent.adversary]
            adversaries = [agent for agent in world.agents if agent.adversary]
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
            rew = rew / self.num_agents
            r1 = 8.03
            r2 = 0.36
            rew = (rew - r2/25) / (r1 - r2)
        elif self.task.type == 4:
            agents = [agent for agent in world.agents if not agent.adversary]
            adversaries = [agent for agent in world.agents if agent.adversary]
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew -= 10
            rew = rew / self.num_agents
            r1 = -11.93
            r2 = -110.29
            rew = (rew - r2/25) / (r1 - r2)

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
