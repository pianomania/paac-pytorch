import argparse
from paac import train

parser = argparse.ArgumentParser(description='parameters_setting')
parser.add_argument('--lr', type=float, default=0.00025, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers(default: 4)')
parser.add_argument('--num-envs', type=int, default=4, metavar='W',
                    help='number of environments a worker holds(default: 4)')
parser.add_argument('--n-steps', type=int, default=5, metavar='NS',
                    help='number of forward steps in PAAC (default: 5)')
parser.add_argument('--env-name', default='BreakoutDeterministic-v4', metavar='ENV',
                    help='environment to train on (default: BreakoutDeterministic-v4)')
parser.add_argument('--max-train-steps', type=int, default=500000, metavar='MS',
                    help='max training step to train PAAC (default: 500000)')
parser.add_argument('--clip-grad-norm', type=int, default=3.0, metavar='MS',
                    help='globally clip gradient norm(default: 3.0)')
parser.add_argument('--record', type=bool, default=False, metavar='R',
                    help='record scores of every environment (default: False)')


if __name__ == "__main__":

    args = parser.parse_args()
    train(args)