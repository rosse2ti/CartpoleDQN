import gym
import numpy as np

# create environment
env = gym.make("Taxi-v3")

# initialize Q-table
Q = np.zeros([env.observation_space.n, env.action_space.n])

# set hyperparameters
gamma = 0.1
alpha = 0.1
epsilon = 0.1

# initialize reward
reward = 0

# initialize environment
state = env.reset()

epochs = 0
total_epochs = 0
episodes = 10000
epsilon_decay = 0.99  # decay factor

for episode in range(episodes):
    epochs = 0
    reward = 0
    epsilon = epsilon * epsilon_decay  # decay step
    state = env.reset()

    #    #####  BASELINE AGENT  #####
    #
    #    while reward != 20:
    #        state, reward, done, info = env.step(env.action_space.sample())
    #        epochs += 1
    #
    #    total_epochs += epochs

    # print("Average timesteps taken: {}".format(total_epochs / episodes))

    # while dropoff state has not been reached
    while reward != 20:
        epochs += 1

        if np.random.rand() < epsilon:
            # exploration option
            action = env.action_space.sample()
        else:
            # exploitation option
            action = np.argmax(Q[state])

        # obtain reward and next state resulting from taking action
        next_state, reward, done, info = env.step(action)

        # update Q-value for state-action pair
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        # update state
        state = next_state

    total_epochs += epochs

print("Average timesteps taken: {}".format(total_epochs / episodes))
