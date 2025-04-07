---
title: "Implementing the Deep Q learning algorithm from scratch"
layout: "single"
date: 2025-03-09
tags: ["machine learning"]
codeMaxLines: 100
---





Deep Q-Learning is one of the foundational algorithms in reinforcement learning, combining the power of deep neural networks with Q-learning to tackle complex decision-making tasks. Unlike traditional Q-learning, which struggles with high-dimensional state spaces, Deep Q-Learning uses neural networks to approximate the Q-value function, enabling it to handle environments with vast or continuous state spaces.

In this tutorial, we will implement Deep Q-Learning *from scratch*. By the end of this guide, you'll gain a deep understanding of:
- How Q-learning works in principle and why deep neural networks are essential for scaling up the algorithm.
- The key components of Deep Q-Learning, such as experience replay and target networks, and how they stabilize learning.
- The step-by-step process to build and train a Deep Q-Learning agent in Python.

But why is it worth learning how to implement Deep Q-Learning? This algorithm is a gateway to many advanced reinforcement learning techniques like Double Q-Learning, Dueling DQN, and methods used in modern-day AI research. It's also a highly practical skill for tackling real-world problems like game-playing AI (e.g., solving Atari or mastering board games), robotics, and dynamic resource allocation.

Whether you're exploring reinforcement learning for research or building powerful AI systems for real-world applications, mastering Deep Q-Learning is an essential step.

Let’s dive in and build Deep Q-Learning from scratch!

## States, Actions, and Episodes in reinforcement learning
In reinforcement learning (RL), _states_, _actions_, and _episodes_ are foundational concepts that describe how an agent interacts with its environment to learn and achieve its goals. Here's a brief explanation of each:
#### **What is a State?**
A **state** represents the agent's current status or observation within the environment. It contains all the essential information that the agent needs to decide its next action.
- Definition: A state \(S_t\) at time \(t\) is the agent's understanding of "where it is" in the environment.
- **Examples:**
    - In a game, the state might include the player's location, the position of enemies, or the score.
    - In real-world applications, states might involve sensor readings, such as a robot's speed or position.

- **State Space:** The set of all possible states that the agent may encounter is referred to as the _state space_. For high-dimensional or continuous state spaces, neural networks are often employed to approximate state representations.

#### **What is an Action?**
An **action** is the decision made by the agent to interact with the environment and transition from one state to another.
- Definition: An action \(A_t\) is selected by the agent at time \(t\) based on a policy \(\pi\), which maps observed states \(S_t\) to actions \(A_t\).
- **Examples:**
    - In a game, an action could be "move left," "jump," or "shoot."
    - In robotics, an action might involve steering a vehicle or adjusting joint angles.

- **Action Space:** The set of all possible actions is called the _action space_, which can be:
    - **Discrete:** A finite set of actions (e.g., moving up, down, left, or right).
    - **Continuous:** An infinite range of actions (e.g., setting a steering angle or adjusting forces).

#### **What is an Episode?**
An **episode** is a complete sequence of agent-environment interactions, starting from the initial state until the task ends or a termination condition is met.
- Definition: An episode consists of a sequence of states, actions, and rewards: \( S_0, A_0, R_1, S_1, A_1, R_2, \dots, S_T \), where \( T \) marks the terminal state or the end of the episode.
- **Examples:**
    - In a board game, an episode ends when the game is won, lost, or drawn.
    - In tasks like navigation, an episode ends when the goal is reached or a time limit is exceeded.

## Deep-Q learning

Deep Q learning is a value-based reinforcement learning method. This means that we learn a  **value function** \(V(s)\) that maps each state to an expected reward starting from state \(s\).
Our goal with reinforcement learning is to get a policy that takes optimal steps. By learning a value function, we will be able to
define a policy.

$$V(s) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t R_t \,\Big|\, S_0 = s \right]$$

In the value function, we always use a \(\gamma\) discounting factor. You might notice that the farther we look into the future,
the larger effect the discounting factor has. Rewards received in the distant future are often uncertain. 
Situations may change, actions may not behave as expected, and the environment dynamics may vary. 
Gamma allows us to give less weight to these uncertain, far-off rewards while focusing more on rewards that are closer and more predictable.
If the value function could perfectly predict what total reward we get starting from a certain state, we would have a
perfect policy that can always take the best action.
To put it in mathematical terms:

$$\pi^*(s) = \arg\max_{a} Q^*(s, a)$$

where \(Q^*(s, a)\) is the expected reward starting from state \(s\), and action \(a\) is taken starting from this state.
\(\pi^*\) is the optimal policy functions that always takes the action that leads to the maximum reward in the rest of the trajectory.
The definition of \(Q\) and \(V\) are very similar. Both are called value-functions, since they return the expected value of the total rewards starting from some state.
\(V\) is called the state-value function, and \(Q\) is called the action-value function.
If we look at the above definition of \(V(s)\), we see that it is the sum of an unknown number of discounted rewards. 
An episode can be potentially infinite, since we don't know how long it lasts.
This means that we cannot program the calculation of this formula in code.
The solution is to rewrite the definition so that it has a finite number of operands in the summation. This is where the **Bellman Equation** comes into play.
The Bellman Equation is derived from the principle of optimality and provides a recursive definition of the value function. For a deeper explanation of its origins and derivation, you can refer to [this article on the Bellman Equation](https://www.datacamp.com/tutorial/bellman-equation-reinforcement-learning) or any comprehensive reinforcement learning resource.

$$V_{\pi}(s) = \mathbb{E} [R_{t+1} + \gamma V_{\pi}(s_{t+1}) | s_t = s]$$

Since this formula contains an expected value operator, we cannot directly use it to update \(V(s)\). Let's rewrite it using the sampled states we might observe during an episode:

$$V_{\pi}(s) \approx \frac{1}{N} \sum_{i=1}^{N} \left( R_{t+1}^{(i)} + \gamma V_{\pi}(s_{t+1}^{(i)}) \right)$$

Here \(N\) is the number of samples in a single episode. In order to implement an agent that learns continously, 
we cannot just use the right side of the above equation to overwrite \(V_{\pi}(s)\). 
Otherwise, the agent would completely forget everything previously learned after each episode. Instead, we 
want to use an update rule that helps our agent to learn over time. The **Temporal Difference** (TD) update rule is a 
key concept in reinforcement learning that combines ideas from Monte Carlo methods and Dynamic Programming. 
It updates value estimates based on the difference (or error) between consecutive predictions, 
allowing learning to occur without waiting for the end of an episode. 
For a more detailed explanation, you can refer to 
[this article on Temporal Difference learning](https://www.lancaster.ac.uk/stor-i-student-sites/jordan-j-hood/2021/04/12/reinforcement-learning-temporal-difference-td-learning/) or other RL texts.
The following is the update formula for the TD approach:

$$V(s) \leftarrow V(s) + \alpha \left( R_{t+1} + \gamma V(s_{t+1}) - V(s) \right)$$

Here, \(\alpha\) is the **learning rate**.
With the TD update rule, we can learn continously during each episode.
We would be able to learn the expected reward starting from each possible state.
But don't forget, our goal is to actually help an agent take an action that will lead to the highest expected cumulative reward.
We need to use a slightly different function called the action-value function that we alluded to earlier. 
It is \(Q(s,a)\), where \(s\) is the current state, \(a\) is an action
the agent can take, and \(Q(s,a)\) should be the total expected reward when we take action \(a\) in state \(s\). 
With this function we can predict the future rewards for each action taken.

We can change the value function in the TD update rule to use the action-value function:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left( R_{t+1} + \gamma max_{a'} Q(s_{t+1},a') - Q(s,a) \right)$$

Let's pause a moment to discuss what each part of this formula represents.

* \(Q(s,a)\) is the original prediction of the expected reward before the update.
* \(R_{t+1}\) is the reward the agent receives when entering the next state from state \(s\).
* \(R_{t+1} + \gamma max_{a'} Q(s_{t+1},a')\) is the estimated total reward starting from state \(s\). This is our target when learning \(Q(s,a)\).
* \(R_{t+1} + \gamma max_{a'} Q(s_{t+1},a') - Q(s,a)\) is the TD error.

We could represent the value function as a table (or a Python dictionary in a concrete implementation) where we map each state to the estimated value.
Indeed, this is exactly how classical Q-learning works.
We could then update the value function at each state using the Bellman equation after each episode.
There are however a few problems with this. In most real-world applications, there are a huge number of states.
Imagine an agent that has a camera as input with a resolution of 1000x1000 pixels and 256 distinct value for each pixel (assuming grayscale images).
We would need to store \(256^{1000x1000}\) distinct values in our table. This is completely unfeasible.
Additionally, for very similar states we would expect to have a similar value estimation. When we update a single value of our
mapping, it does not affect the estimation for any other state, it cannot *generalize*.

These issues are addressed by using a deep neural network as an estimator instead of a simple mapping.
In deep learning, we specify how a network learns by defining a loss function. How can you derive a loss function from the Bellman equation ?
We do this by simply taking the TD error of the original update rule:

$$R_{t+1} + \gamma max_{a'} Q(s_{t+1},a') - Q(s,a)$$

This will be our Q-learning loss.

Putting all of this together, we get a basic version of the Deep Q-learning:

![](images/deep_q_tutorial/basic_deep_q_learning.jpg)

You might notice that we are using a slightly different notation for the action-value function with a new \(\theta\)
parameter. This represents the learnable parameters of our neural network that we will update.

Now we are ready to implement the simplest version of the algorithm.

## Implementation

We are going to use the [montaincar task](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/)  from OpenAI Gym as a testbed for our Deep-Q algorithm implementation.
The environment consists of a car positioned between two hills, and the goal is to drive the car up the steep slope 
on the right-hand side. However, the car's engine is not powerful enough to directly climb the hill in one go. Instead, 
the agent must learn to build momentum by moving back and forth, using gravity to its advantage.
The goal is to get the car to the flag positioned at the top of the right hill (a specific position in the state space). 
Once the car reaches this position, the episode ends successfully.
The agent receives a reward of `-1` for every time step it takes until it reaches the goal. 
This encourages the agent to reach the goal in as few steps as possible.
Please take some time to study the documentation of this environment.

First, let's import everything we will need.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
from collections import deque

import imageio
```

We will implement the \(Q(s, a; \theta)\) function as a neural network. The tensor \(\theta\) is the collection of
all parameters inside the network that we train.
The exact number of layers and parameters are assigned empirically, based on the agent's performance. 
That is another way of saying that I tried different number of layers and parameters until this configuration seemed
to be the best. I will leave experimenting with different architectures as a homework for the reader.

```python
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

Next, we define our hyperparameters. 

```python
num_episodes = 200
max_steps = 100
learning_rate = 0.001
gamma = 0.99
```

We initialize the environment, the neural network and the optimizer.

```python
env = gym.make("MountainCar-v0")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_network = QNetwork(state_dim, action_dim)
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()
```


Next, we are ready to implement the actual training loop. This works
exactly like we showed with the pseudocode. We will also save the total
rewards per episode. If everything works as expected, we should see the
episodic rewards increasing over time.

```python
epsiode_rewards = []
for episode in tqdm(range(num_episodes)):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    total_reward = 0

    for step in range(max_steps):
        q_values = q_network(state)
        action = torch.argmax(q_values).item()  # Best action

        # Take action
        next_state, reward, done, _, _ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        total_reward += reward

        # Compute target and prediction
        target = reward
        if not done:
            target += gamma * torch.max(q_network(next_state)).item()
        target = torch.tensor([target], dtype=torch.float32)

        q_value = q_network(state)[0, action]
        loss = loss_fn(q_value, target)

        # Update weights using backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        if done:
            break
    episode_rewards.append(total_reward)
env.close()
```

Code for plotting the rewards.

```python
plt.figure(figsize=(10, 6))
plt.plot(episode_rewards, label='Episode Reward')
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Episode Reward', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
```

![](images/deep_q_tutorial/reward1.png)

The reward is not getting better over time. What is happening?
To get out of the valley and reach the flag, the cart needs to build up sufficient momentum.
To do this, it needs to accelerate for multiple steps in one direction, then in the other, and then switch to the other direction again... 
If this is what is happening, we should see it in the decisions that an agent makes.
Lets visualize the decisions that the agent is making in each episode.

![](images/deep_q_tutorial/actions_heatmap.png)

What do we see here?
The color of each pixel represents what action the agent is taking each step of each episode.
"0" is pushing the cart to the left, and "1" for the cart to the right. 
In the first few episodes we see a lot of seemingly random actions. This is because in the beginning, we initialize the weights of the
value network randomly. If everything worked as expected we should be seeing long stretches of left/right action as the agent tries to gain momentum.
Instead we see just "1" almost constantly during the later episodes.
Why is this?
Remember that the value network tries to predict the total rewards starting from the current state. If the agent never successfully reaches the flag on the top of the mountain, then
the network will learn that the expected reward of the trajectory starting from all states is the same. Which means that from
the agent's point of view, no action is better than any other.
The agent cannot get out of the valley until it learns better reward predictions, and it cannot learn better reward predictions until it gets out of the valley. 
This seems like a chicken-or-egg problem.
The traditional solution for this problem is to occasionally take random actions so that the agent has a chance to reach better states by accident.
We also want to rely less on random actions as the network learns to correctly predict the rewards. We called this an **Epsilon-greedy** policy.
With a probability of **epsilon**, we take a random action. After each episode, we decrease the epsilon value a little bit.

We will use some new hyperparameters.

```python
epsilon = 1.0
epsilon_decay = 0.99
epsilon_min = 0.01
```

Here, **epsilon** will be the probability of choosing a random action instead of using the network.
**epsilon decay** is used to decrease the current value of epsilon after each episode. As the agent learns, we want to
rely less on random actions. **epsilon min** is the minimum value of epsilon. We might not want epsilon to reach zero.
There is some possibility that even a well-trained agent might get stuck, and an occasional random action can help it in
these cases.

This is how the implementation of our new policy will look like:

```python
if np.random.rand() < epsilon:
    # Random action
    action = env.action_space.sample()  
else:
    # Greedy action
    q_values = q_network(state)
    action = torch.argmax(q_values).item()
```

At the end of each episode, we decrease the epsilon parameter:

```python
if epsilon > epsilon_min:
    epsilon *= epsilon_decay
```

With these changes, let's redo the training and visualize the rewards.

![](images/deep_q_tutorial/reward2.png)

The rewards are still not getting better over time.
We would expect that as the agent learns, it would start to find higher reward states, and the total reward per episode would be higher over time.
But we can see that the rewards per episode remain at \(-100\).
Remember that in this environment we get a reward of \(-1\) at every step, and we set the maximum number of steps at \(100\). This means that
the agent never reaches the flag, so there is no positive experience to learn from.
In other reinforcement learning environments such as [Cart pole](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
we get positive rewards for every step where we manage to
keep the pole upright. 
In mountaincar, we only get a better total reward if the cart manages to reach the flag.
If the flag is never reached, no real learning will take place. 
This isn't easy in the beginning, because we have to do a lot of random actions in the right order to get out of the valley.
If the maximum number of steps per episode is too low, the agent will not have the chance to reach the flag via random actions.
So let's increase the maximum number of actions per episode from \(100\) to \(1000000\), and retrain the agent.

```python
max_steps = 1000000
```

![](images/deep_q_tutorial/reward3.png)

This is a much better result than previously.
The agent actually learns how to get out of the valley. 
If you did the training, you might have noticed that the first few episodes take a very long time, since the agent relies on random chance
to get out of the valley. As it learns more it takes less time for it to reach the flag.
Let's visualize one episode with the trained agent.

```python
q_network.load_state_dict(torch.load("best_q_network.pth"))
env = gym.make("MountainCar-v0", render_mode="rgb_array")

state, _ = env.reset()
state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

frames = []
for _ in range(10000):
    q_values = q_network(state)
    action = torch.argmax(q_values).item()  # Best action

    # Take action
    next_state, _, done, _, _ = env.step(action)
    next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

    state = next_state
    
    frame = env.render()
    frames.append(frame)
    if done:
        break

env.close()
```

![](images/deep_q_tutorial/mountaincar_good.gif)

We can see that the car easily builds up momentum and escapes the valley.
So that would be the Deep Q-learning algorithm. This is a very basic implementation that does not use
techniques such as **experience replay** to make the training more efficient. I wanted to
leave out such extras to make it as easy as possible to understand this technique.

For those who just want to see the final result, I will leave the implementation here.


```python
num_episodes = 200
max_steps = 1000000
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.99
epsilon_min = 0.01

env = gym.make("MountainCar-v0")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_network = QNetwork(state_dim, action_dim)
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

best_reward_sum = float('-inf')  # Best rolling 10-episode reward sum
reward_window = deque(maxlen=10)  # Sliding window for last 10 episode rewards

best_model_path = "best_q_network.pth"  # File to save the best model weights

# Training loop
for episode in tqdm(range(num_episodes)):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    total_reward = 0

    for _ in range(max_steps):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = q_network(state)
            action = torch.argmax(q_values).item()

        # Take action
        next_state, reward, done, _, _ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        total_reward += reward

        # Compute target and prediction
        target = reward
        if not done:
            target += gamma * torch.max(q_network(next_state)).item()
        target = torch.tensor([target], dtype=torch.float32)

        q_value = q_network(state)[0, action]
        loss = loss_fn(q_value, target)

        # Update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        if done:
            break


    # Check if this is the best network
    if len(reward_window) == reward_window.maxlen:
        reward_sum = sum(reward_window)
        if reward_sum > best_reward_sum:
            best_reward_sum = reward_sum
            torch.save(q_network.state_dict(), best_model_path)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

env.close()

plt.figure(figsize=(10, 6))
plt.plot(, label='Episode Reward')
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Episode Reward', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
```


