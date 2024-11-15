import random
from types import SimpleNamespace as SN

import numpy as np

from .core import Action


def onehot(x, n):
    return [1 if i == x else 0 for i in range(n)]


def binary(x, n):
    return [int(i) for i in bin(x)[2:].zfill(n)]


def TaskGenerator(num_agent=None, type=None):
    task = {}

    if num_agent is None:
        num_agent = random.randint(3, 5)
    num_adversary = 0
    num_landmark = 0
    shape = 0

    if type is None:
        type = random.randint(0, 4)

    if type == 0: # occupy
        num_landmark = num_agent
    elif type == 1: # form
        num_landmark = 1
        # shape = random.randint(1, 3)
        shape = 1
    elif type == 2: # push
        num_agent = 3
        num_adversary = num_agent
        num_landmark = num_agent
    elif type == 3: # tag1 追捕：4个追2个
        num_agent = 4
        num_adversary = 2
        num_landmark = 2
    elif type == 4: # tag2 被追捕：2个躲4个
        num_agent = 2
        num_adversary = 4
        num_landmark = 2
    elif type == 5:
        num_agent = 5
        num_adversary = 5
        num_landmark = 5

    if type < 5:
        code = binary(num_agent, 3) + onehot(type, 5) + onehot(shape, 4) # 3 + 5 + 4 = 12
    else:
        code = binary(num_agent, 3) + [1,1,1,1,1] + onehot(shape, 4)

    task["num_agents"] = num_agent
    task["num_adversaries"] = num_adversary
    task["num_landmarks"] = num_landmark
    task["type"] = type
    task["shape"] = shape

    task["code"] = code

    task = SN(**task)
    return task


def action_callback_push_adversary(agent, world):
    id = agent.name.split(" ")[-1]
    id = int(id)
    target_landmark = world.landmarks[id]
    target_pos = target_landmark.state.p_pos
    agent_pos = agent.state.p_pos
    del_pos = target_pos - agent_pos
    a = Action()
    a.u = np.zeros(world.dim_p)
    if abs(del_pos[0]) > abs(del_pos[1]):
        if del_pos[0] > 0:
            a.u[0] = 1.
            a.u[1] = 0.
        else:
            a.u[0] = -1.
            a.u[1] = 0.
    else:
        if del_pos[1] > 0:
            a.u[0] = 0.
            a.u[1] = 1.
        else:
            a.u[0] = 0.
            a.u[1] = -1.
    sensitivity = 5
    a.u *= sensitivity
    return a


def action_callback_tag1_adversary(agent, world):
    # 被追捕
    agent_pos = agent.state.p_pos
    # 找到最近的好agent
    dists = [np.sqrt(np.sum(np.square(agent_pos - a.state.p_pos))) 
             for a in world.agents if not a.adversary]
    target_id = np.argmin(dists)
    target_pos = world.agents[target_id].state.p_pos
    del_pos = target_pos - agent_pos
    a = Action()
    a.u = np.zeros(world.dim_p)
    if abs(del_pos[0]) < abs(del_pos[1]):
        if del_pos[0] > 0:
            a.u[0] = -1.
            a.u[1] = 0.
        else:
            a.u[0] = 1.
            a.u[1] = 0.
    else:
        if del_pos[1] > 0:
            a.u[0] = 0.
            a.u[1] = -1.
        else:
            a.u[0] = 0.
            a.u[1] = 1.
    sensitivity = 5
    a.u *= sensitivity
    return a


def action_callback_tag2_adversary(agent, world):
    # 追捕
    agent_pos = agent.state.p_pos
    # 找到最近的好agent
    dists = [np.sqrt(np.sum(np.square(agent_pos - a.state.p_pos))) 
             for a in world.agents if not a.adversary]
    target_id = np.argmin(dists)
    target_pos = world.agents[target_id].state.p_pos
    del_pos = target_pos - agent_pos
    a = Action()
    a.u = np.zeros(world.dim_p)
    if abs(del_pos[0]) > abs(del_pos[1]):
        if del_pos[0] > 0:
            a.u[0] = 1.
            a.u[1] = 0.
        else:
            a.u[0] = -1.
            a.u[1] = 0.
    else:
        if del_pos[1] > 0:
            a.u[0] = 0.
            a.u[1] = 1.
        else:
            a.u[0] = 0.
            a.u[1] = -1.
    sensitivity = 5
    a.u *= sensitivity
    return a


def action_callback_push_away_adversary(agent, world):
    id = agent.name.split(" ")[-1]
    id = int(id)
    a = Action()
    a.u = np.zeros(world.dim_p)
    a.u[0] = 0.
    a.u[1] = 0.
    return a