import numpy as np
import torch as th
import torch.distributions as tdist
from gym import spaces

from .basic_controller import BasicMAC


# This multi-agent controller shares parameters between agents
class CQMixMAC(BasicMAC):
    def select_actions(
        self,
        ep_batch,
        t_ep,
        t_env,
        bs=slice(None),
        test_mode=False,
        past_actions=None,
        critic=None,
        target_mac=False,
        explore_agent_ids=None,
    ):
        chosen_actions = self.cem_sampling(ep_batch, t_ep, bs)

        if not test_mode:  # do exploration
            act_noise = 0.1
            x = chosen_actions.clone().zero_()
            chosen_actions += act_noise * x.clone().normal_()

        return chosen_actions

    def forward(
        self,
        ep_batch,
        t,
        return_hs=True,
        actions=None,
        hidden_states=None,
        select_actions=False,
        test_mode=False,
    ):
        agent_inputs = self._build_inputs(ep_batch, t)
        ret = self.agent(agent_inputs, self.hidden_states, actions=actions)
        if select_actions:
            self.hidden_states = ret["hidden_state"]
            return ret
        agent_outs = ret["Q"]
        self.hidden_states = ret["hidden_state"]

        if self.agent_output_type == "pi_logits":
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                agent_outs = (
                    1 - self.action_selector.epsilon
                ) * agent_outs + th.ones_like(
                    agent_outs
                ) * self.action_selector.epsilon / agent_outs.size(
                    -1
                )
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), self.hidden_states.view(ep_batch.batch_size, self.n_agents, -1)

    def _build_inputs(self, batch, t, target_mac=False, last_target_action=None):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av

        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions"][:, t]))
            else:
                inputs.append(batch["actions"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device)
                .unsqueeze(0)
                .expand(bs, -1, -1)
            )

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)

        return inputs

    # def _get_input_shape(self, scheme):
    #     input_shape = scheme["obs"]["vshape"]
    #     if self.args.obs_last_action:
    #         if getattr(self.args, "discretize_actions", False):
    #             input_shape += scheme["actions_onehot"]["vshape"][0]
    #         else:
    #             input_shape += scheme["actions"]["vshape"][0]
    #     if self.args.obs_agent_id:
    #         input_shape += self.n_agents

    #     return input_shape

    def cem_sampling(self, ep_batch, t, bs, critic=None):
        # Number of samples from the param distribution
        N = 64
        # Number of best samples we will consider
        Ne = 6

        ftype = (
            th.FloatTensor
            if not next(self.agent.parameters()).is_cuda
            else th.cuda.FloatTensor
        )
        mu = ftype(ep_batch[bs].batch_size, self.n_agents, self.args.n_actions).zero_()
        std = (
            ftype(ep_batch[bs].batch_size, self.n_agents, self.args.n_actions).zero_()
            + 1.0
        )
        its = 0

        maxits = 2
        agent_inputs = self._build_inputs(ep_batch[bs], t)
        hidden_states = self.hidden_states.reshape(
            -1, self.n_agents, self.args.rnn_hidden_dim
        )[bs].repeat(N, 1, 1, 1)

        # Use feed-forward critic here, so it takes only the obs input
        critic_inputs = []
        if critic is not None:
            critic_inputs.append(ep_batch[bs]["obs"][:, t])
            critic_inputs = th.cat(
                [
                    x.reshape(ep_batch[bs].batch_size * self.n_agents, -1)
                    for x in critic_inputs
                ],
                dim=1,
            )

        while its < maxits:
            dist = tdist.Normal(
                mu.view(-1, self.args.n_actions), std.view(-1, self.args.n_actions)
            )
            actions = dist.sample((N,)).detach()
            actions_prime = th.tanh(actions)

            if critic is None:
                ret = self.agent(
                    agent_inputs.unsqueeze(0)
                    .expand(N, *agent_inputs.shape)
                    .contiguous()
                    .view(-1, agent_inputs.shape[-1]),
                    hidden_states if hidden_states is not None else self.hidden_states,
                    actions=actions_prime.view(-1, actions_prime.shape[-1]),
                )
                out = ret["Q"].view(N, -1, 1)
            else:
                out, _ = critic(
                    critic_inputs.unsqueeze(0)
                    .expand(N, *critic_inputs.shape)
                    .contiguous()
                    .view(-1, critic_inputs.shape[-1]),
                    actions=actions_prime.view(-1, actions_prime.shape[-1]),
                )
                out = out.view(N, -1, 1)

            topk, topk_idxs = th.topk(out, Ne, dim=0)
            mu = th.mean(
                actions.gather(0, topk_idxs.repeat(1, 1, self.args.n_actions).long()),
                dim=0,
            )
            std = th.std(
                actions.gather(0, topk_idxs.repeat(1, 1, self.args.n_actions).long()),
                dim=0,
            )
            its += 1

        topk, topk_idxs = th.topk(out, 1, dim=0)
        action_prime = th.mean(
            actions_prime.gather(0, topk_idxs.repeat(1, 1, self.args.n_actions).long()),
            dim=0,
        )
        chosen_actions = (
            action_prime.clone()
            .view(ep_batch[bs].batch_size, self.n_agents, self.args.n_actions)
            .detach()
        )

        return chosen_actions
