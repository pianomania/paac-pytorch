import torch
from torch.multiprocessing import Process
from atari import atari
import numpy as np
from utils import Recorder
import time

class worker(Process):

    def __init__(self, worker_id, num_env, game_name, n_stack,
                 wait_queue, act_queue):
        super(worker, self).__init__()

        self.worker_id = worker_id
        self.num_env = num_env
        self.n_stack = n_stack

        self.wait_queue = wait_queue
        self.act_queue = act_queue

        self.envs = []
        self.index_base = worker_id * num_env
        self.episode_length = [0] *  num_env
        #self.recorder = []

        for i in range(num_env):
            time.sleep(0.1)
            access_index = self.index_base+i
            env = atari(game_name, n_stack)
            env.reset()
            self.envs.append(env)

            #self.recorder.append(Recorder(int(worker_id*num_env+i), game_name))

    def run(self):
        
        super(worker, self).run()

        while True:

            action = self.act_queue.get()

            if action is None:
                break

            send_state = np.zeros((self.num_env, self.n_stack, 84, 84))
            send_reward = np.zeros(self.num_env)
            send_mask = np.zeros(self.num_env)

            for i, env in enumerate(self.envs):

                access_index = self.index_base+i

                a = action[i]
                s, r, terminal, real_terminal = env.step(a)

                #self.recorder[i].record(r, real_terminal)
                self.episode_length[i] += 1

                if real_terminal or self.episode_length[i] > 10000:
                    env.reset()
                    s = env._state
                    self.episode_length[i] = 0
                    terminal = real_terminal = True

                send_state[i] = s
                send_reward[i] = r
                send_mask[i] = 0 if terminal else 1

            self.wait_queue.put((send_state, send_reward, send_mask))
