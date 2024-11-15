import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.n_transf_mixer import TransformerMixer
from modules.mixers.n_transf_mixer_task import \
    TransformerMixer as TransformerMixerTask
import torch as th
from torch.optim import RMSprop, Adam
from utils.th_utils import get_parameters_num


class CQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda  else 'cpu')
        self.params = list(mac.parameters())


        if args.mixer == "transf_mixer":
            self.mixer = TransformerMixer(args)
        elif args.mixer == "transf_mixer_task":
            self.mixer = TransformerMixerTask(args)
        else:
            raise "mixer error"
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params,  lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0

        # priority replay
        self.use_per = getattr(self.args, 'use_per', False)
        self.return_priority = getattr(self.args, "return_priority", False)
        if self.use_per:
            self.priority_max = float('-inf')
            self.priority_min = float('inf')

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        actions = batch["actions"][:, :-1]
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        chosen_action_qvals = []
        mac_hs = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):  # Note the minimum value of max_seq_length is 2
            agent_outs, mac_h = self.mac.forward(batch, actions=batch["actions"][:, t:t + 1].detach(), t=t)
            chosen_action_qvals.append(agent_outs)
            mac_hs.append(mac_h)
        chosen_action_qvals = th.stack(chosen_action_qvals[:-1], dim=1)  # Concat over time
        mac_hs = th.stack(mac_hs[:-1], dim=1)  # Concat over time

        best_target_actions = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            action_outs = self.target_mac.select_actions(batch, t_ep=t, t_env=None, test_mode=True)
            best_target_actions.append(action_outs)
        best_target_actions = th.stack(best_target_actions, dim=1)  # Concat over time
        target_max_qvals = []
        target_mac_hs = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, target_mac_h = self.target_mac.forward(batch, t=t, actions=best_target_actions[:, t].detach())
            target_max_qvals.append(target_agent_outs)
            target_mac_hs.append(target_mac_h)
        target_max_qvals = th.stack(target_max_qvals[1:], dim=1)  # Concat over time
        target_mac_hs = th.stack(target_mac_hs[1:], dim=1)  # Concat over time

        # Mix
        # chosen_action_qvals = self.mixer(chosen_action_qvals.view(-1, self.args.n_agents, 1), batch["state"][:, :-1])
        # target_max_qvals = self.target_mixer(target_max_qvals.view(-1, self.args.n_agents, 1), batch["state"][:, 1:])
        # chosen_action_qvals = chosen_action_qvals.view(batch.batch_size, -1, 1)
        # target_max_qvals = target_max_qvals.view(batch.batch_size, -1, 1)

        chosen_action_qvals_ = chosen_action_qvals
        hyper_weights = self.mixer.init_hidden().expand(batch.batch_size, self.args.n_agents, -1)
        chosen_action_qvals = th.zeros(batch.batch_size, batch.max_seq_length-1, 1).to(self.args.device)
        for t in range(batch.max_seq_length - 1):
            mixer_out, hyper_weights = self.mixer(
                chosen_action_qvals_[:, t].view(-1, 1, self.args.n_agents),
                mac_hs[:, t,].detach(),
                hyper_weights,
                batch["state"][:, t],
                batch["obs"][:, t])
            chosen_action_qvals[:, t] = mixer_out.squeeze(-1)
        
        target_max_qvals_ = target_max_qvals
        hyper_weights = self.target_mixer.init_hidden().expand(batch.batch_size, self.args.n_agents, -1)
        target_max_qvals = th.zeros(batch.batch_size, batch.max_seq_length-1, 1).to(self.args.device)
        for t in range(batch.max_seq_length - 1):
            target_mixer_out, hyper_weights = self.target_mixer(
                target_max_qvals_[:, t].view(-1, 1, self.args.n_agents), # (batch, 1, n_agents)
                target_mac_hs[:, t],
                hyper_weights,
                batch["state"][:, t],
                batch["obs"][:, t]
            )
            target_max_qvals[:, t] = target_mixer_out.squeeze(-1)

        # Calculate 1-step Q-Learning targets
        targets = rewards.expand_as(target_max_qvals) + self.args.gamma * (1 -
                                                        terminated.expand_as(target_max_qvals)) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = L_td = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", L_td.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

        info = {}
        return info

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
            
    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))