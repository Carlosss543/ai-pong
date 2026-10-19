# AI Pong

Simple code for training a neural network to play Atari Pong with
[Gymnasium](https://gymnasium.farama.org/) and PyTorch. The agent learns from
the game screen and rewards using a REINFORCE-style policy-gradient update.

## Demo

**Episode 800**

https://github.com/user-attachments/assets/6c3228e0-3bd0-4615-86fc-6cb2618283f0

Our model controls the right-hand side paddle.

## Setup and training

Dependencies: PyTorch, Gymnasium with Atari support (ALE), and MoviePy for video recording.

```bash
pip install torch "gymnasium[atari]" ale-py moviepy
```

To launch a training run :

```bash
python main.py
```

Settings are at the top of `main.py`. Set `resume_training = True` to load an
existing checkpoint. Checkpoints are saved every 100 episodes; videos are
recorded every 200 episodes.

## REINFORCE

For a fixed observation $s$, the policy $\pi_\theta(a \mid s)$ gives the
probability of choosing action $a$. Let $R(s,a)$ be its expected reward.

The goal of training is to minimize the negative expected reward:

$$
L(\theta) = -\mathbb{E}_{a \sim \pi_\theta(\cdot \mid s)}[R(s,a)]
$$

For that, we compute the following gradient:

$$
\begin{aligned}
\nabla_\theta L(\theta)
&= -\nabla_\theta \sum_a R(s,a) \pi_\theta(a \mid s) \\
&= -\sum_a R(s,a)\,\nabla_\theta \pi_\theta(a \mid s) \\
&= -\sum_a R(s,a) \pi_\theta(a \mid s) \nabla_\theta \log \pi_\theta(a \mid s) \\
&= -\mathbb{E}_{a \sim \pi_\theta(\cdot \mid s)}
   [R(s,a)\,\nabla_\theta \log \pi_\theta(a \mid s)]
\end{aligned}
$$

Here we use $\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta$.
For an action $a$ sampled from the policy and its observed reward $r$,
the sampled surrogate loss is $-r\log \pi_\theta(a \mid s)$:
positive rewards make the sampled action more likely, negative rewards less likely.

## Discounted rewards

Actions are weighted by future rewards, discounted with $\gamma = 0.99$:

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
$$

The code stops this sum at each scored point, then normalizes the returns to
obtain the normalized discounted rewards $\widehat{G}_t$.

Over the $N$ collected steps, we average the log probabilities of sampled
actions, weighted by their normalized discounted returns:

$$
\widehat{L}_{\mathrm{surrogate}}(\theta)
= -\frac{1}{N}\sum_{t=0}^{N-1}
\widehat{G}_t\,\log \pi_\theta(a_t \mid s_t).
$$

In the code, this corresponds to:

```python
loss = -(torch.cat(log_probs_list) * discounted_rewards).mean()
```
