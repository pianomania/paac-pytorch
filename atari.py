import gym
import cv2
import numpy as np


def _process_frame84(frame):
    frame = cv2.resize(frame, (84, 84))
    frame *= (1.0 / 255.0)
    return frame


class atari(object):
    
    def __init__(self, game_name, n_stack=4):

        self.env = gym.make(game_name) 
        self.valid_action = self.env.env._action_set
        self.n_stack = n_stack
        self._state = np.zeros((n_stack, 84, 84))

        self.reset()
        self.lives = self.env.env.ale.lives()


    def reset(self):
        
        s = self.env.reset()
        s = self.env.env.ale.getScreenGrayscale().squeeze().astype('float32')
        s = _process_frame84(s)
        self.update_state(s)
        self.lives = self.env.env.ale.lives()

    
    def step(self, action):

        _, r, terminal, info = self.env.step(action)
        screen = self.env.env.ale.getScreenGrayscale().squeeze().astype('float32')
        screen = _process_frame84(screen)

        if terminal:
            self._state *= 0.0
        else:
            self.update_state(screen)

        if self.lives > info['ale.lives'] and info['ale.lives'] > 0:
            pesudo_terminal = True
            self.lives = info['ale.lives']
        else:
            pesudo_terminal = terminal

        return self._state, r, pesudo_terminal, terminal


    def update_state(self, obs):
        self._state[:-1] = self._state[1:]
        self._state[-1] =  obs
