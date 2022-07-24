from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.autograd as ag
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from torch import autograd
from rlkit.torch.pytorch_util import PiecewiseSchedule, ConstantSchedule
from rlkit.core import logger
from collections import deque

class AlgoTrainer(TorchTrainer):
	def __init__(
			self,
			env,
			env_name,
			policy,
			qfs, #list of q functions
			target_qfs, #list of target q-functions

			qloss_version,
			qpen_version,
			qtarget_version,
			#VAE, #gradpen additions
			LAMBDA=5, #added

			discount=0.99,
			reward_scale=1.0,

			policy_lr=1e-3,
			qf_lr=1e-3,
			optimizer_class=optim.Adam,

			soft_target_tau=1e-2,
			plotter=None,
			render_eval_paths=False,

			use_automatic_entropy_tuning=True,
			target_entropy=None,
			policy_eval_start=0,
			ucb_ratio=0.01,
			ensemble=8,
			min_weight_ood=0.2,
			decay_factor=1.01,
			prior=False,

			# CQL
			min_q_version=3, #khong quan trong, dung cho CQL
			temp=1.0, #khong quan trong, dung cho CQL
			min_q_weight=1.0, #khong quan trong, dung cho CQL

			# sort of backup
			max_q_backup=False,
			deterministic_backup=True,
			num_random=10,
			with_lagrange=False,
			lagrange_thresh=5.0,  # CQL new end
			policy_name = "VACG_3",
 
	):
		super().__init__()
		# assert env_name in ["halfcheetah-random-v2", "halfcheetah-medium-v2", "halfcheetah-expert-v2", "halfcheetah-medium-expert-v2",
		# 	"halfcheetah-medium-replay-v2", "walker2d-random-v2", "walker2d-medium-v2", "walker2d-expert-v2",
		# 	"walker2d-medium-expert-v2", "walker2d-medium-replay-v2", "hopper-random-v2", "hopper-medium-v2",
		# 	"hopper-expert-v2", "hopper-medium-expert-v2", "hopper-medium-replay-v2"]
		self.env_name = env_name
		self.policy_name = policy_name
		self.env = env
		self.policy = policy
		self.qfs = qfs

		#self.vae = vae #gradpen additions
		self.LAMBDA = LAMBDA
		self.qloss_version = qloss_version
		self.qpen_version = qpen_version
		self.qtarget_version = qtarget_version

		self.target_qfs = target_qfs
		self.soft_target_tau = soft_target_tau  # 0.005

		self.use_automatic_entropy_tuning = use_automatic_entropy_tuning

		# define an optimizer for log_alpha. The initial value of log_alpha is 0.
		if self.use_automatic_entropy_tuning:                 # True
			if target_entropy:
				self.target_entropy = target_entropy
			else:                                             # use this
				self.target_entropy = -np.prod(self.env.action_space.shape).item()  # -6.
			self.log_alpha = ptu.zeros(1, requires_grad=True)  # [0.]
			self.alpha_optimizer = optimizer_class([self.log_alpha], lr=policy_lr)  # policy_lr=0.0001

		self.with_lagrange = with_lagrange               # True or False
		if self.with_lagrange:
			self.target_action_gap = lagrange_thresh     # 5.0
			self.log_alpha_prime = ptu.zeros(1, requires_grad=True)  # [0.]
			# Optimizer for log_alpha_prime
			self.alpha_prime_optimizer = optimizer_class([self.log_alpha_prime], lr=qf_lr)  # qf_lr=0.0003

		self.plotter = plotter  # None
		self.render_eval_paths = render_eval_paths  # False

		self.qf_criterion_all = nn.MSELoss(reduction='none')
		self.qf_criterion = nn.MSELoss()
		self.vf_criterion = nn.MSELoss()

		self.discount = discount                 # 0.99
		self.reward_scale = reward_scale         # 1
		self.eval_statistics = OrderedDict()     # dict
		self._n_train_steps_total = 0
		self._need_to_update_eval_statistics = True

		self._current_epoch = 0
		self._policy_update_ctr = 0
		self._num_q_update_steps = 0
		self._num_policy_update_steps = 0
		self._num_policy_steps = 1

		self.temp = temp                         # 1.0
		self.min_q_version = min_q_version       # 3
		self.min_q_weight = min_q_weight         # 1

		self.softmax = torch.nn.Softmax(dim=1)  #
		self.softplus = torch.nn.Softplus(beta=self.temp, threshold=20)

		self.max_q_backup = max_q_backup  # False
		self.deterministic_backup = deterministic_backup  # True
		self.num_random = num_random  # 10

		# For implementation on the discrete env
		self.discrete = False

		# ucb
		self.ensemble = ensemble
		self.ucb_ratio = ucb_ratio
		self.prior = prior
		logger.log(f"Ensemble: {self.ensemble}, UCB ratio of offline data: {self.ucb_ratio}")

		# Define optimizer for critic and actor
		self.policy_optimizer = optimizer_class(self.policy.parameters(), lr=policy_lr)
		self.qfs_optimizers = []
		for i in range(self.ensemble):      # each ensemble member has its optimizer
			self.qfs_optimizers.append(optimizer_class(self.qfs[i].parameters(), lr=qf_lr))

		# record previous-Q for adjust the Q penalty of ood actions. (or remove it.)
		self.previous_Q1 = deque(maxlen=5)
		self.previous_Q2 = deque(maxlen=20)

		if self.env_name in ["halfcheetah-expert-v2", "hopper-expert-v2", "walker2d-expert-v2"]:
			# Constant for Expert is better since the dataset is optimal, and OOD actions are useless. We can always use a large penalty.
			self.w_schedule = ConstantSchedule(2.0)
			logger.log("w_schedule = ConstantSchedule(2.0)")
		else:
			self.w_schedule = PiecewiseSchedule([(0, 5.0), (50000, 1.0)], outside_value=1.0)
			self.min_weight_ood = min_weight_ood
			self.decay_factor = decay_factor
			logger.log("w_schedule = PiecewiseSchedule([(0, 5.0), (50000, 1.0)], outside_value=1.0)")
			logger.log(f"min_weight_ood: {self.min_weight_ood}, reduce {self.decay_factor}")

		logger.log(f"\n\n *********\n PBRL Algorithm\n*********")

	def _get_tensor_values(self, obs, actions, network=None):
		action_shape = actions.shape[0]                      # 2560
		obs_shape = obs.shape[0]                             # 256
		num_repeat = int(action_shape / obs_shape)           # 10
		obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])  # （2560, obs_dim）
		preds = network(obs_temp.cuda(), actions.cuda())     # 输入 Q-network, 返回 (2560, 1)
		preds = preds.view(obs.shape[0], num_repeat, 1)      # (256, 10, 1)
		return preds

	def _get_policy_actions(self, obs, num_actions, network=None):
		# obs.shape=(256, obs_dim), num_actions=10. After repeat, obs_temp.shape=(2560, 10)
		obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
		# new_obs_actions=(2560, act_dim), new_obs_log_pi.shape=(2560, 1)
		new_obs_actions, _, _, new_obs_log_pi, *_ = network(obs_temp, reparameterize=True, return_log_prob=True)

		if not self.discrete:       # new_obs_actions.shape=(2560, act_dim), new_obs_log_pi.shape=(256, 10, 1)
			return new_obs_actions, new_obs_log_pi.view(obs.shape[0], num_actions, 1)
		else:
			return new_obs_actions

	def ucb_func(self, obs, act, mean=False):
		# Using the main-Q network to calculate the bootstrapped uncertainty
		# Sample 10 ood actions for each obs, so the obs should be expanded before calculating
		action_shape = act.shape[0]                          # 2560
		obs_shape = obs.shape[0]                             # 256
		num_repeat = int(action_shape / obs_shape)           # 10
		if num_repeat != 1:
			obs = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])  # （2560, obs_dim）
		# Bootstrapped uncertainty
		q_pred = []
		for i in range(self.ensemble):
			q_pred.append(self.qfs[i](obs.cuda(), act.cuda()))
		ucb = torch.std(torch.hstack(q_pred), dim=1, keepdim=True)   # (2560, 1)
		assert ucb.size() == (obs.size()[0], 1)
		if mean:
			q_pred = torch.mean(torch.hstack(q_pred), dim=1, keepdim=True)
		return ucb, q_pred

	def ucb_func_target(self, obs_next, act_next):
		# Using the target-Q network to calculate the bootstrapped uncertainty
		# Sample 10 ood actions for each obs, so the obs should be expanded before calculating
		action_shape = act_next.shape[0]             # 2560
		obs_shape = obs_next.shape[0]                # 256
		num_repeat = int(action_shape / obs_shape)   # 10
		if num_repeat != 1:
			obs_next = obs_next.unsqueeze(1).repeat(1, num_repeat, 1).view(obs_next.shape[0] * num_repeat, obs_next.shape[1])  # （2560, obs_dim）
		# Bootstrapped uncertainty
		target_q_pred = []
		for i in range(self.ensemble):
			target_q_pred.append(self.target_qfs[i](obs_next.cuda(), act_next.cuda()))
		ucb_t = torch.std(torch.hstack(target_q_pred), dim=1, keepdim=True)
		assert ucb_t.size() == (obs_next.size()[0], 1)
		return ucb_t, target_q_pred
	
	def cal_gradpen(self, QNet, real_actions, generated_actions, corresponding_obs, center=0, alpha=None, LAMBDA=1):
		if alpha is not None:
			alpha = torch.tensor(alpha).cuda()  # torch.rand(real_data.size(0), 1, device=device)
		else:
			alpha = torch.rand(real_actions.size(0), 1).cuda()
		alpha = alpha.expand(real_actions.size())

		interpolated_actions = alpha * real_actions + ((1 - alpha) * generated_actions)

		interpolated_actions.requires_grad_(True)
		corresponding_obs.requires_grad_(True)

		QNet_interpolates = QNet(corresponding_obs.cuda(), interpolated_actions.cuda())

		gradients = ag.grad(outputs=QNet_interpolates, inputs=[corresponding_obs, interpolated_actions],
							grad_outputs=torch.ones(QNet_interpolates.size()).cuda(),
							create_graph=True, retain_graph=True, only_inputs=True)[0]
		#print(f"gradients: {gradients}")
		gradient_penalty = ((gradients.norm(2, dim=1) - center) ** 2).mean() * LAMBDA
		return gradient_penalty

	#make a tensor duplicate num_repeat times current tensor
	def _duplicate_tensor(self, tensor, num_repeat):
		return tensor.unsqueeze(1).repeat(1, num_repeat, 1).view(tensor.shape[0] * num_repeat, tensor.shape[1])

	def weighted_mse(self, input_tensor, target_tensor, weight_tensor):
		return torch.sum((input_tensor - target_tensor) ** 2 * weight_tensor) / torch.sum(weight_tensor)

	def train_from_torch(self, batch):
		self._current_epoch += 1
		rewards = batch['rewards']             # shape=(256, 1)
		terminals = batch['terminals']         # shape=(256, 1)
		obs = batch['observations']            # shape=(256, 17)
		actions = batch['actions']             # shape=(256, 6)
		next_obs = batch['next_observations']  # shape=(256, 17)
		batch_size = rewards.size()[0]
		action_dim = actions.size()[-1]
		""" Policy and Alpha Loss
		"""
		# batch data 的 obs 通过当前 policy 得到 new_obs_actions
		new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
										obs, reparameterize=True, return_log_prob=True)

		if self.use_automatic_entropy_tuning:
			alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
			self.alpha_optimizer.zero_grad()
			alpha_loss.backward()
			self.alpha_optimizer.step()
			alpha = self.log_alpha.exp()
		else:
			alpha_loss = 0
			alpha = 1

		"""
		QF Loss
		"""
		new_next_actions, _, _, new_log_pi, *_ = self.policy(next_obs, reparameterize=True, return_log_prob=True)
		new_curr_actions, _, _, new_curr_log_pi, *_ = self.policy(obs, reparameterize=True, return_log_prob=True)

		# compute ucb for (s,a) and (s',a')
		ucb_current, q_pred = self.ucb_func(obs, actions)
		ucb_next, target_q_pred = self.ucb_func_target(next_obs, new_next_actions)

		self._num_q_update_steps += 1

		# Sample OOD actions (the random ood action is only used for evaluation)
		random_actions_tensor = torch.FloatTensor(batch_size * self.num_random, actions.shape[-1]).uniform_(-1, 1)  # num_actions=10
		curr_actions_tensor, curr_log_pis = self._get_policy_actions(obs, num_actions=self.num_random, network=self.policy)
		new_curr_actions_tensor, new_log_pis = self._get_policy_actions(next_obs, num_actions=self.num_random, network=self.policy)

		# calculate the uncertainty for OOD actions
		ucb_rand, qf_rand_all = self.ucb_func(obs, random_actions_tensor)
		ucb_curr_actions, qf_curr_actions_all = self.ucb_func(obs, curr_actions_tensor)
		ucb_next_actions, qf_next_actions_all = self.ucb_func(next_obs, new_curr_actions_tensor)


		assert ucb_rand.size() == ucb_curr_actions.size() == ucb_next_actions.size() == qf_rand_all[0].size() == \
			qf_curr_actions_all[0].size() == qf_next_actions_all[0].size() == (2560, 1)

		# weight of ood penalty
		weight_of_ood_l2 = self.w_schedule.value(self._num_q_update_steps)

		for qf_index in np.arange(self.ensemble):
			#experiments already run:
				#PBRL original: no weighted, pessimistic Q_loss (target = 1, qloss = 1, qpen=3)  
				#PBRL_OUR: weighted no pessimistic Q_loss + gradients penalty + no ood penalty
				#VACG2: pessimistic target + ood penalty + gradients penalty
				#VACG_3: target_version = 2, qloss_version = 3, qpen_version = 2
			# Q-target
			if (self.qtarget_version == 1): #pessimistic target
				q_target = self.reward_scale*rewards + (1. - terminals)*self.discount*(target_q_pred[qf_index] - self.ucb_ratio*ucb_next)
			elif (self.qtarget_version == 2): #target khong pessimistic
				target_scalar_q_value = torch.min(qf_next_actions_all[qf_index], dim=1)[0]
				q_target = self.reward_scale*rewards + (1. - terminals)*self.discount*target_scalar_q_value
			else:
				exit("Q-target version not supported")
			q_target = q_target.detach()

			# Critic loss. MSE. The input shape is (256,1)
			if (self.qloss_version == 1): #MSE loss
				qf_loss_in = self.qf_criterion(q_pred[qf_index], q_target)
			elif (self.qloss_version == 2): #MSE weighted by variance of Q
				qf_loss_in = self.weighted_mse(q_pred[qf_index], q_target, torch.max(ucb_current.detach(), torch.Tensor([1]).expand_as(ucb_current).cuda())).cuda()
			elif (self.qloss_version == 3): #MSE weighted by variance + CQL penalty 
				qf_loss_in_weighted_mse = self.weighted_mse(q_pred[qf_index], q_target, torch.max(ucb_current.detach(), torch.Tensor([1]).expand_as(ucb_current).cuda())).cuda()
				random_actions_tensor = torch.FloatTensor(q_pred[qf_index].shape[0] * self.num_random, actions.shape[-1]).uniform_(-1, 1)    # num_actions=10
				q_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qfs[qf_index])   # (256, 10, 1)
				q_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.qfs[qf_index])  # (256, 10, 1)
				q_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.qfs[qf_index])  # (256, 10, 1)
				cat_q = torch.cat([q_rand, q_pred[qf_index].unsqueeze(1), q_next_actions, q_curr_actions], 1)   # (256, 31, 1)
				#std_q = torch.std(cat_q, dim=1)

				random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])  
				cat_q = torch.cat([q_rand - random_density,
                 					q_next_actions - new_log_pis.detach(), q_curr_actions - curr_log_pis.detach()], 1)
				min_qf_loss = torch.logsumexp(cat_q / self.temp, dim=1,).mean() * self.min_q_weight * self.temp

				min_qf_loss = min_qf_loss - q_pred[qf_index].mean() * self.min_q_weight
				if self.with_lagrange:
					alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
					min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)
					min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)

					self.alpha_prime_optimizer.zero_grad()
					alpha_prime_loss = (-min_qf1_loss - min_qf2_loss)*0.5 
					alpha_prime_loss.backward(retain_graph=True)
					self.alpha_prime_optimizer.step()
				qf_loss_in = qf_loss_in_weighted_mse + min_qf_loss
			else: 
				exit("Q-loss version not supported")

			duplicated_cur_obs = self._duplicate_tensor(obs, self.num_random)
			gaussian_noise = torch.normal(mean=0, std=0.11, size=curr_actions_tensor.shape).cuda()
			curr_noisy_actions_tensor = curr_actions_tensor + gaussian_noise

			if (self.qpen_version == 1) or (self.qpen_version == 3): 
				cat_qf_ood = torch.cat([qf_curr_actions_all[qf_index], qf_next_actions_all[qf_index]], 0)
				assert cat_qf_ood.size() == (2560*2, 1)

				cat_qf_ood_target = torch.cat([
					torch.maximum(qf_curr_actions_all[qf_index] - weight_of_ood_l2 * ucb_curr_actions, torch.zeros(1).cuda()),
					torch.maximum(qf_next_actions_all[qf_index] - 0.1 * ucb_next_actions, torch.zeros(1).cuda())], 0)
				cat_qf_ood_target = cat_qf_ood_target.detach()
				assert cat_qf_ood_target.size() == (2560*2, 1)
				qf_pen_ood_pbrl = self.qf_criterion(cat_qf_ood, cat_qf_ood_target)

			qf_pen_ood_grad = self.cal_gradpen(	self.qfs[qf_index],
											curr_actions_tensor, 
											curr_noisy_actions_tensor.detach(), 
											duplicated_cur_obs.detach(), 
											center=0, LAMBDA=self.LAMBDA, alpha=None) #naiive option
			# Final loss
			if (self.qpen_version == 1): #with ood penalty
				qf_loss = qf_loss_in + qf_pen_ood_pbrl + qf_pen_ood_grad
			elif (self.qpen_version == 2):	#no odd penalty
				qf_loss = qf_loss_in + qf_pen_ood_grad
			elif (self.qpen_version == 3):	#odd, no grad penalty
				qf_loss = qf_loss_in + qf_pen_ood_pbrl
			else: 
				exit("Q-pen version not supported")

			# Update the Q-functions
			self.qfs_optimizers[qf_index].zero_grad()
			qf_loss.backward(retain_graph=True)
			self.qfs_optimizers[qf_index].step()

		# Actor loss
		q_new_actions_all = []
		for i in range(self.ensemble):
			q_new_actions_all.append(self.qfs[i](obs, new_obs_actions))
		q_new_actions = torch.min(torch.hstack(q_new_actions_all), dim=1, keepdim=True).values
		assert q_new_actions.size() == (batch_size, 1)

		policy_loss = (alpha * log_pi - q_new_actions).mean()

		self._num_policy_update_steps += 1
		self.policy_optimizer.zero_grad()
		policy_loss.backward()
		self.policy_optimizer.step()

		# Soft update the target-Nets
		for i in np.arange(self.ensemble):
			if self.prior:
				ptu.soft_update_from_to(self.qfs[i].main_network, self.target_qfs[i].main_network, self.soft_target_tau)
			else:
				ptu.soft_update_from_to(self.qfs[i], self.target_qfs[i], self.soft_target_tau)

		# Save some statistics for eval
		if self._need_to_update_eval_statistics:
			# record the previous Q to update the OOD weight. (or remove it)
			if "medium-v2" in self.env_name or "medium-replay-v2" in self.env_name or "random" in self.env_name:
				new_record_q = torch.stack(qf_curr_actions_all, dim=-1).mean().cpu().detach().numpy()
				if self._num_q_update_steps > 50000 and weight_of_ood_l2 > self.min_weight_ood and np.mean(self.previous_Q1) < np.mean(np.array(self.previous_Q2)[:-5]):
					self.w_schedule = ConstantSchedule(weight_of_ood_l2/self.decay_factor)
					logger.log(f"Lower Penalty by {self.decay_factor}, current-Q:{np.mean(self.previous_Q1)}, Previous Q:{np.mean(np.array(self.previous_Q2)[:-5])}, new weight: {weight_of_ood_l2/self.decay_factor}")
				self.previous_Q1.append(new_record_q)
				self.previous_Q2.append(new_record_q)

			# for record
			with torch.no_grad():
				ucb_rand, q_rand = self.ucb_func(obs, torch.distributions.uniform.Uniform(-1, 1).sample((batch_size, action_dim)), mean=True)
				ucb_noise1, q_noise1 = self.ucb_func(obs, actions + torch.normal(mean=torch.zeros((batch_size, action_dim)), std=torch.ones((batch_size, action_dim)) * 0.1).cuda(), mean=True)
				ucb_noise2, q_noise2 = self.ucb_func(obs, actions + torch.normal(mean=torch.zeros((batch_size, action_dim)), std=torch.ones((batch_size, action_dim)) * 0.5).cuda(), mean=True)
				ucb_noise3, q_noise3 = self.ucb_func(obs, actions + torch.normal(mean=torch.zeros((batch_size, action_dim)), std=torch.ones((batch_size, action_dim)) * 1.0).cuda(), mean=True)

			self._need_to_update_eval_statistics = False
			"""
			Eval should set this to None.
			This way, these statistics are only computed for one batch.
			"""
			# self.eval_statistics['UCB weight of CurrPolicy'] = weight_of_ood_l2
			self.eval_statistics['Q CurrPolicy'] = np.mean(ptu.get_numpy(qf_curr_actions_all[0]))
			self.eval_statistics['Q RandomAction'] = np.mean(ptu.get_numpy(qf_rand_all[0]))
			self.eval_statistics['Q NextAction'] = np.mean(ptu.get_numpy(qf_next_actions_all[0]))

			self.eval_statistics['UCB CurrPolicy'] = np.mean(ptu.get_numpy(ucb_curr_actions))
			self.eval_statistics['UCB RandomAction'] = np.mean(ptu.get_numpy(ucb_rand))
			self.eval_statistics['UCB NextAction'] = np.mean(ptu.get_numpy(ucb_next_actions))

			self.eval_statistics['Q Offline'] = np.mean(ptu.get_numpy(q_pred[0]))
			self.eval_statistics['Q Noise1'] = np.mean(ptu.get_numpy(q_noise1))
			self.eval_statistics['Q Noise2'] = np.mean(ptu.get_numpy(q_noise2))
			self.eval_statistics['Q Noise3'] = np.mean(ptu.get_numpy(q_noise3))
			self.eval_statistics['Q Rand'] = np.mean(ptu.get_numpy(q_rand))

			self.eval_statistics['UCB Offline'] = np.mean(ptu.get_numpy(ucb_current))
			self.eval_statistics['UCB Next'] = np.mean(ptu.get_numpy(ucb_next))
			self.eval_statistics['UCB Noise1'] = np.mean(ptu.get_numpy(ucb_noise1))
			self.eval_statistics['UCB Noise2'] = np.mean(ptu.get_numpy(ucb_noise2))
			self.eval_statistics['UCB Noise3'] = np.mean(ptu.get_numpy(ucb_noise3))
			self.eval_statistics['UCB Rand'] = np.mean(ptu.get_numpy(ucb_rand))

			self.eval_statistics['QF Loss in'] = np.mean(ptu.get_numpy(qf_loss_in))
			#self.eval_statistics['QF Loss ood'] = np.mean(ptu.get_numpy(qf_loss_ood))
			self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))

			self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(policy_loss))
			self.eval_statistics.update(create_stats_ordered_dict('Q Targets', ptu.get_numpy(q_target)))
			self.eval_statistics.update(create_stats_ordered_dict('Log Pis', ptu.get_numpy(log_pi)))

			if self.use_automatic_entropy_tuning:
				self.eval_statistics['Alpha'] = alpha.item()
				self.eval_statistics['Alpha Loss'] = alpha_loss.item()

		self._n_train_steps_total += 1

	def get_diagnostics(self):
		return self.eval_statistics

	def end_epoch(self, epoch):
		self._need_to_update_eval_statistics = True

	@property
	def networks(self):
		base_list = [
			self.policy,
			self.qfs,
			self.target_qfs,
		]
		return base_list

	def get_snapshot(self):
		return dict(
			policy=self.policy,
			qfs=self.qfs,
			target_qf1=self.target_qfs,
		)