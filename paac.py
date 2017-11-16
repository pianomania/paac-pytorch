import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
from torch.multiprocessing import Process, Pipe

from worker import worker
from model import paac_ff

import gym
import numpy as np

def train(args):

	torch.multiprocessing.set_start_method('forkserver')
	
	num_envs = args.num_envs
	num_workers = args.num_workers
	total_envs = num_workers * num_envs
	game_name = args.env_name
	max_train_steps = args.max_train_steps
	n_steps = args.n_steps
	init_lr = args.lr
	gamma = args.gamma
	clip_grad_norm = args.clip_grad_norm
	num_action = gym.make(game_name).action_space.n
	image_size = 84
	n_stack = 4

	model = paac_ff(min_act=num_action).cuda()

	x = Variable(torch.zeros(total_envs, n_stack, image_size, image_size), volatile=True).cuda()
	xs = [Variable(torch.zeros(total_envs, n_stack, image_size, image_size)).cuda() for i in range(n_steps)]

	share_reward = [Variable(torch.zeros(total_envs)).cuda() for _ in range(n_steps)]
	share_mask = [Variable(torch.zeros(total_envs)).cuda() for _ in range(n_steps)]
	constant_one = torch.ones(total_envs).cuda()

	optimizer = optim.Adam(model.parameters(), lr=init_lr)

	workers = []
	parent_conns = []
	child_conns = []
	for i in range(num_workers):
		parent_conn, child_conn = Pipe()
		w = worker(i, num_envs, game_name, n_stack, child_conn, args)
		w.start()
		workers.append(w)
		parent_conns.append(parent_conn)
		child_conns.append(child_conn)

	new_s = np.zeros((total_envs, n_stack, image_size, image_size))

	for global_step in range(1, max_train_steps+1):

		cache_v_series = []
		entropies = []
		sampled_log_probs = []

		for step in range(n_steps):
			
			xs[step].data.copy_(torch.from_numpy(new_s))
			v, pi = model(xs[step])
			cache_v_series.append(v)

			sampling_action = pi.data.multinomial(1)

			log_pi = (pi+1e-12).log()
			entropy = -(log_pi*pi).sum(1)
			sampled_log_prob = log_pi.gather(1, Variable(sampling_action)).squeeze()
			sampled_log_probs.append(sampled_log_prob)
			entropies.append(entropy)
			
			send_action = sampling_action.squeeze().cpu().numpy()
			send_action = np.split(send_action, num_workers)

			# send action and then get state
			for parent_conn, action in zip(parent_conns, send_action):
				parent_conn.send(action)
			
			batch_s, batch_r, batch_mask = [], [], []
			for parent_conn in parent_conns:
				s, r, mask = parent_conn.recv()
				batch_s.append(s)
				batch_r.append(r)
				batch_mask.append(mask)

			new_s = np.vstack(batch_s)
			r = np.hstack(batch_r).clip(-1, 1) # clip reward
			mask = np.hstack(batch_mask)

			share_reward[step].data.copy_(torch.from_numpy(r))
			share_mask[step].data.copy_(torch.from_numpy(mask))

		x.data.copy_(torch.from_numpy(new_s))
		v, _ = model(x) # v is volatile
		R = Variable(v.data.clone())
		v_loss = 0.0
		policy_loss = 0.0
		entropy_loss = 0.0
		
		for i in reversed(range(n_steps)):

			R =  share_reward[i] + 0.99*share_mask[i]*R
			advantage = R - cache_v_series[i]
			v_loss += advantage.pow(2).mul(0.5).mean()

			policy_loss -= sampled_log_probs[i].mul(advantage.detach()).mean()
			entropy_loss -= entropies[i].mean()
		
		total_loss = policy_loss + entropy_loss.mul(0.02) +  v_loss*0.5
		total_loss = total_loss.mul(1/(n_steps))

		# adjust learning rate
		new_lr = init_lr - (global_step/max_train_steps)*init_lr
		for param_group in optimizer.param_groups:
			param_group['lr'] = new_lr
		
		optimizer.zero_grad()
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), clip_grad_norm)

		optimizer.step()

		if global_step % 10000 == 0 :
			torch.save(model.state_dict(), './model/model_%s.pth' % game_name)

	for parent_conn in parent_conns:
		parent_conn.send(None)

	for w in workers:
		w.join()
