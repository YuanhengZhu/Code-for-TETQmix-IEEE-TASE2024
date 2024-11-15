import numpy as np

from ..core import Agent, Landmark, World
from ..scenario import BaseScenario
from ..utils import TaskGenerator, action_callback_push_adversary as action_callback


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        world.dim_c = 2

        task = TaskGenerator(type=1)
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
        self.task = task
        self.num_agents = task.num_agents
        self.num_adversaries = task.num_adversaries
        self.num_landmarks = task.num_landmarks
        self.num_entities = self.num_agents + self.num_adversaries + self.num_landmarks
        # add agents
        world.agents = [Agent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            agent.adversary = False
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
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35])
        # random properties for landmarks
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

        self.reset_targets(world, task)

    def reset_targets(self, world, task):
        self.targets = []
        num_agents = task.num_agents
        shape = task.shape
        center = world.landmarks[0].state.p_pos
        # 1 正多边形
        # 2 ─
        # 3 └
        d = 1
        if shape == 1:
            for i in range(num_agents):
                delta = d * np.array([np.cos(2 * np.pi * i / num_agents), np.sin(2 * np.pi * i / num_agents)])
                self.targets.append(center + delta)
        elif shape == 2:
            for i in range(num_agents):
                delta = d * (-1)**i * ((i+1)//2) * np.array([1, 0])
                self.targets.append(center + delta)
        elif shape == 3:
            for i in range(num_agents):
                if i % 2 == 0:
                    delta = d * ((i+1)//2) * np.array([1, 0])
                else:
                    delta = d * ((i+1)//2) * np.array([0, 1])
                self.targets.append(center + delta)
        for i, target in enumerate(self.targets): # for visualization
            world.landmarks.append(Landmark())
            world.landmarks[-1].name = 'target %d' % i
            world.landmarks[-1].collide = False
            world.landmarks[-1].movable = False
            world.landmarks[-1].size = 0.05
            world.landmarks[-1].color = np.array([0, 1, 0])
            world.landmarks[-1].state.p_pos = target
            world.landmarks[-1].state.p_vel = np.zeros(world.dim_p)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        if agent.name != "agent 0":
            return 0

        rew = 0
      
        for t in self.targets:
            dist = [np.sqrt(np.sum(np.square(a.state.p_pos - t)))
                    for a in world.agents]
            rew -= min(dist)
            
        for agent in world.agents:
            if agent.collide:
                for a in world.agents:
                    if a is not agent and self.is_collision(a, agent):
                        rew -= 1

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
        for landmark in world.landmarks[:self.num_landmarks]:
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
        for landmark in world.landmarks[:self.num_landmarks]:
            is_self = 0.
            is_agent = 0.
            is_adversary = 0.
            is_landmark = 1.
            pos = landmark.state.p_pos - pos_a
            vel = landmark.state.p_vel
            feats[idx:idx+self.entity_obs_feats] = task_code + [pos[0], pos[1], vel[0], vel[1], is_self, is_agent, is_adversary, is_landmark] + [1.]
            idx += self.entity_obs_feats

        return feats
