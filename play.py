from atari import atari
import gym
import argparse
from model import paac_ff
from torch.autograd import Variable
import torch
import numpy as np

parser = argparse.ArgumentParser(description='parameters_setting')
parser.add_argument('--env-name', default='BreakoutDeterministic-v4', metavar='ENV',
                    help='environment to train on (default: BreakoutDeterministic-v4)')

if __name__ == "__main__":

    args = parser.parse_args()
    n_stack = 4
    image_size = 84

    env = atari(args.env_name)
    num_action = env.env.action_space.n

    model = paac_ff(num_action).cuda()
    model.load_state_dict(torch.load("./model/model_"+args.env_name+".pth"))

    env.reset()

    x = Variable(torch.zeros(1, n_stack, image_size, image_size), volatile=True).cuda()

    x.data.copy_(torch.from_numpy(env._state[np.newaxis,:,:,:]))

    terminal = False

    while terminal is not True:

        v, pi = model(x)
        sampling_action = pi.data.multinomial(1).squeeze().cpu().numpy()
        next_state, reward, _, terminal = env.step(sampling_action)
        env.env.render()
        x.data.copy_(torch.from_numpy(next_state))


