---
title: "Implementing the Reinforce algorithm from scratch"
layout: "single"
date: 2025-01-08
tags: ["machine learning"]
codeMaxLines: 100
---

One of the best ways to understand Reinforcement Learning (RL) is to implement its algorithms from scratch. 
While libraries like [Stable Baselines](https://stable-baselines3.readthedocs.io/en/master/) offer powerful tools for real-world applications, 
custom implementations allow you to see how these methods work under the hood.

This article walks you through implementing **Reinforce**, a fundamental algorithm in the family of [Policy Gradient Methods](https://spinningup.openai.com/en/latest/algorithms/vpg.html). 
Reinforce is a natural starting point: its simplicity makes it approachable, on the other hand it serves as the basis for
some of the most commonly used Reinforcement Learning algorithms like 
[Actor-Critic Method](https://spinningup.openai.com/en/latest/algorithms/vpg.html) and 
[Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/vpg.html).

By the end of this guide, you’ll not only have a working implementation but also a stronger intuition for how policy gradient methods optimize decision-making in dynamic environments.

## The objective function

The goal of policy gradient methods is to maximize the value of an objective function.

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \Big[ R(\tau) \Big]$$

where 

$$\tau = (s_0, a_0, r_1, \dots, s_T, a_T, r_{T+1})$$ is a variable representing a trajectory for an episode as a list
of state,action,reward triplets.

We can expand the expected value in the definition:

$$
J(\theta) = \sum_{\tau} P(\tau | \pi_\theta) R(\tau), \quad
$$

where

$$
P(\tau | \pi_\theta) = P(s_0) \prod_{t=0}^{T} \pi_\theta(a_t | s_t) P(s_{t+1} | s_t, a_t)
$$

is the probability of taking this trajectory with a given policy. In policy gradient methods this policy is represented
by a deep network that we optimize using gradient descent, or one of its derived methods like Adam.

## Policy Gradient Theorem

The problem with the above definition is that we cannot directly implement it in code, because we don't
know the probability functions. This is where the Policy Gradient Theorem comes into play

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \Bigg[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) R_t \Bigg],
$$

where

$$
R_t = \sum_{k=t}^T \gamma^{k-t} r_{k+1}.
$$

For a proof of the theorem, you can check **Section 13.2** of [Reinforcement Learning:
An Introduction](http://incompleteideas.net/book/bookdraft2017nov5.pdf).
We cannot calculate the above expected value directly, since that would imply that we have access to all possible trajectories.
Instead, we can play for a number of episodes and calculate an estimation based on these samples:

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^T R_t^{(i)} \nabla_\theta \log \pi_\theta(a_t^{(i)} | s_t^{(i)}),
$$

We can calculate this gradient and update the policy with gradient **ascent**.

$$
\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)
$$

Now we can implement all of this directly in Pytorch. Before we start, there are two changes to the formula to make.
We are not going to implement the gradient update manually, but use one of the builtin optimizers in Pytorch.
For this reason we will need to calculate the objective function, and not it's gradient directly. Secondly,
since the builtin optimizers use gradient **descent**, we will negate the formula to make it work.
Here is what the final function looks like:

$$
-\frac{1}{N} \sum_{i=1}^N \sum_{t=0}^T R_t^{(i)} \log \pi_\theta(a_t^{(i)} | s_t^{(i)})
$$

## Reinforce algorithm

$$
\begin{aligned}
\textbf{Input:} & \text{ Policy } \pi_\theta, \text{ learning rate } \alpha, \text{ discount factor } \gamma. \\
\textbf{Repeat:} \\
& \quad \text{1. Collect a trajectory } \tau = (s_0, a_0, r_1, \dots, s_T, a_T, r_{T+1}) \text{ using } \pi_\theta. \\
& \quad \text{2. Compute the discounted rewards } R_t = \sum_{k=t}^T \gamma^{k-t} r_{k+1}. \\
& \quad \text{3. Compute the policy gradient:} \\
& \quad \quad \nabla_\theta J(\theta) = \mathbb{E}_\tau \Big[ \sum_{t=0}^T R_t \nabla_\theta \log \pi_\theta(a_t | s_t) \Big]. \\
& \quad \text{4. Update the policy parameters: } \theta \leftarrow \theta + \alpha \nabla_\theta J(\theta). \\
\textbf{Until:} & \text{Policy converges or reaches stopping criterion.}
\end{aligned}
$$

The Reinforce algorithm collects a number of sample episodes or trajectories, calculates the gradient and updates the policy
network. As you can see from above, we update the policy after each episode. This is different from value-based
Reinforcement Learning algorithms like **SARSA** or **Q-learning** where we continuously update the value network
during the episodes.


## Implementation

Let's test the Reinforce algorithm in the [Cartpole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) environment.

We are going to need the following imports:

```python
import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from torch.distributions import Categorical
```

First, we are going to try out a random policy
just to see that the environment works properly.

```python
# The 'render_mode' parameter is required for visuallizing the episode.
env = gym.make("CartPole-v1", render_mode="rgb_array")
env.reset()
frames = []
max_steps = 1000
for step in range(max_steps):
    # Take a random action in the environment.
    action = env.action_space.sample()
    # Execute the selected action.
    _, _, terminated, truncated, info = env.step(action)
    # Collect the frames of the episode for visualization.
    frame = env.render()
    frames.append(frame)
    if terminated:
        break
```

We can visualize the trajectory inside a notebook like so:

```python
for frame in frames:
    plt.imshow(frame)
    clear_output(wait=True)
    plt.axis('off')
    plt.show()
```

We should see a playback similar to this:
![Random Cartpole](images/reinforce_tutorial/cartpole_random.gif)

### Policy

In order to implement the Reinforce algorithm, we will need a policy network.
The policy needs to be able to take a state or observation from the environment
and return an action that will be executed in the environment.

```python
class Policy(nn.Module):
    def __init__(self, state_space_dimensions, action_space_size):
        super(Policy, self).__init__()
        self.fc = nn.Linear(state_space_dimensions, action_space_size)

    def forward(self, x):
        x = self.fc(x)
        return F.softmax(x, dim=1)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        # We always select the most probable action.
        action = torch.argmax(probs)
        return action.item()
```

We will need both the total discounted reward for the episode and 
the log probability of the actions. Let's modify the main loop to record these as well as the policy to return the log probability.

Here is our new policy:

```python
class Policy(nn.Module):
...
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state)
        action = torch.argmax(probs, dim=1)
        # We need to return the log probabilities
        return action.cpu().item(), torch.log(probs[0, action])
```


We are going to implement the Reinforce algorithm piece by piece.
First, to run the game with the policy for a single episode:

```python
state, _ = env.reset()
rewards = []
log_probabilities = []
steps = 0
max_steps = 1000
for step in range(max_steps):
    action, log_prob = policy.act(state)
    log_probabilities.append(log_prob)
    state, reward, terminated, truncated, info = env.step(action)
    # We collect the rewards for the gradient calculation later
    rewards.append(reward)
    steps += 1
    if terminated:
        break
```

After the episode is complete, we calculate the cumulative reward which will be needed
for the objective function.

```python
discounted_rewards = []
cumulative_reward = 0
gamma = 0.99
for reward in reversed(rewards):
    cumulative_reward = reward + gamma * cumulative_reward
    discounted_rewards.insert(0, cumulative_reward)
```

Keep in mind that we need the discounted reward for all timesteps in the episode. This is why this variable is a list
and not a single number. The element 

```python
discounted_rewards[i]
```
represents the total reward we get in the episode starting from time 'i'.

Now that we have the rewards, we can calculate the actual objective function as in the formula.

```python
discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
log_probabilities = torch.cat(log_probabilities)
# We multiply by "-1" so that we can use gradient descent to optimize the policy. 
objective_function = -torch.sum(log_probabilities * discounted_rewards)
```

## Training

Lets put this all together, and make sure to train for several episodes.
We are also going to save the average number of steps that we 'survive' every 100th episode. This way we can see
the progression in the agent's training. We should be seeing this average go up over time if everything works out.

```python
gamma = 0.99
policy = Policy(state_space_dimensions, action_space_size).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
num_episodes = 1000
steps_survived = []


for episode in range(num_episodes):
    state, _ = env.reset()
    rewards = []
    log_probabilities = []
    steps = 0
    max_steps = 1000
    for step in range(max_steps):
        action, log_prob = policy.act(state)
        log_probabilities.append(log_prob)
        state, reward, terminated, truncated, info = env.step(action)
        # We collect the rewards for the gradient calculation later
        rewards.append(reward)
        steps += 1
        if terminated:
            break
    
    # Calculate discounted rewards
    discounted_rewards = []
    cumulative_reward = 0
    for reward in reversed(rewards):  # Reverse the rewards list
        cumulative_reward = reward + gamma * cumulative_reward
        discounted_rewards.insert(0, cumulative_reward)
    
    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
    log_probabilities = torch.cat(log_probabilities)
    objective_function = -torch.sum(log_probabilities * discounted_rewards)
    
    optimizer.zero_grad()
    objective_function.backward()
    optimizer.step()

    steps_survived.append(steps)
```

After running the training, we plot the average number of survived episodes:

```python
running_avg = np.array(steps_survived).reshape((-1, 100)).mean(axis=-1)

plt.figure(figsize=(10, 6))
plt.plot(range(0, len(running_avg) * 100, 100), running_avg, marker='o', linestyle='-', color='b', label='Steps Survived')
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Steps Survived', fontsize=12)
plt.title('Training Progress: Steps survived after training.', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
```

![](images/reinforce_tutorial/train_v1.png)

This is not what we expected. The model is not getting better (by much) with training. What is happening? 
Lets visualize the actions taken in a single episode in each step for the trained policy.
The agent can only nudge the pole left or right, so we visualize this. 

![](images/reinforce_tutorial/trained_decisions.png)

The trained model is always choosing the same action in each step! No wonder the agent is not surviving.
If we nudge the cart in one direction, the pole will definitely fall over.

But why is this happening?
When we first initialize our policy network, it doesn’t know how to make good decisions yet. 
To learn, the agent needs to discover a series of actions that keep the pole upright. 
If you recall, we implemented the policy such that we always choose the action with the highest probability. This makes it deterministic.
With a deterministic policy, the agent keeps repeating the same action.
This behavior creates a fundamental problem in reinforcement learning: 
How can the agent discover good trajectories if it’s stuck repeating the same suboptimal ones?
Earlier I choose a slightly wrong implementation of the policy to demonstrate this point. 
Normally in the Reinforce algorithm implementation we sample from the probability distribution and not choose the most probable action each time. 
This way we introduce stochasticity which can help the agent find better trajectories by adding an exploration element in the decision.

Here is an updated policy implementation:

```python
class Policy(nn.Module):
...
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state)
        # We sample from the probability distribution of actions using PyTorch's 'Categorical' class.
        m = Categorical(probs)
        action = m.sample()
        return action.cpu().item(), m.log_prob(action)
```

Training with the new policy yields the following result:

![](images/reinforce_tutorial/train_v2.png)

This is much better ! We can see that the agent learns to balance the pole better as the training
progresses. Lets visualize an episode:

![Trained Cartpole](images/reinforce_tutorial/cartpole_trained.gif)

Using the new policy, the agent is able to balance the pole almost indefinitely, just as we wanted.
And with that, we have a basic, but working implementation of the Reinforce algorithm.