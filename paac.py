import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
from torch.multiprocessing import Process, Queue

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
	wait_queues = []
	act_queues = []
	for i in range(num_workers):
		wait_queue = Queue()
		act_queue = Queue()
		w = worker(i, num_envs, game_name, n_stack,
                 		  wait_queue, act_queue)
		w.start()
		workers.append(w)
		wait_queues.append(wait_queue)
		act_queues.append(act_queue)
 
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
			for act_queue, action in zip(act_queues, send_action):
				act_queue.put(action)

			get_s, get_r, get_mask = [], [], []
			for wait_queue in wait_queues:
				s, r, mask = wait_queue.get()
				get_s.append(s)
				get_r.append(r)
				get_mask.append(mask)

			new_s = np.vstack(get_s)
			r = np.hstack(get_r).clip(-1, 1) # clip reward
			mask = np.hstack(get_mask)

			share_reward[step].data.copy_(torch.from_numpy(r))
			share_mask[step].data.copy_(torch.from_numpy(mask))

		x.data.copy_(torch.from_numpy(new_s))
		v, pi = model(x) # v and pi is volatile
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

		new_lr = init_lr - (global_step/max_train_steps)*init_lr
		for param_group in optimizer.param_groups:
			param_group['lr'] = new_lr
		
		optimizer.zero_grad()
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), clip_grad_norm)

		optimizer.step()

		if global_step % 10000 == 0 :
			torch.save(model.state_dict(), './model/model_%s.pth' % game_name)

	for act_queue in act_queues:
		act_queue.put(None)

	for w in workers:
		w.join()
