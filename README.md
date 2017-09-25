# paac-pytorch

This is a PyTorch implementation of PAAC from ["Efficient Parallel Methods for Deep Reinforcement Learning"](https://arxiv.org/abs/1705.04862)

![BeamReider](./assets/BeamRider.gif)  ![Breakout](./assets/Breakout.gif)  ![Pong](./assets/Pong.gif)  ![Qbert](./assets/Qbert.gif)

# Usage
- You can train the agent by:

```
python main.py --env-name BreakoutDeterministic-v4 --num-workers 4
```

- You can play the game by:
```
python play.py --env-name BreakoutDeterministic-v4
```

# Results

<p float="first 4 envs">
  <img src="./assets/breakout/env_0.png" width="240" heigh="150">
  <img src="./assets/breakout/env_1.png" width="240" heigh="150">
  <img src="./assets/breakout/env_2.png" width="240" heigh="150">
  <img src="./assets/breakout/env_3.png" width="240" heigh="150">
</p>

# Notes



# References

- [origin paper's open source](https://github.com/Alfredvc/paac)
- [openai's universe-starter-agent](https://github.com/openai/universe-starter-agent)
- [pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c)

