import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Embedding, Reshape
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

env = gym.make("CartPole-v0").env

env.step(1)
print(env.step(1))

action_size = env.action_space.n
state_size = env.observation_space.n
np.random.seed(123)
env.seed(123)

env.reset()
env.step(env.action_space.sample())[0]

model = Sequential()
model.add(Embedding(500, 10, input_length=1))
model.add(Reshape((10,)))
model.add(Dense(50, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(action_size, activation="linear"))
print(model.summary())

memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(
    model=model,
    nb_actions=action_size,
    memory=memory,
    nb_steps_warmup=500,
    target_model_update=1e-2,
    policy=policy,
)
dqn.compile(Adam(learning_rate=1e-3), metrics=["mae"])
dqn.fit(
    env,
    nb_steps=1000,
    visualize=False,
    verbose=1,
    nb_max_episode_steps=10,
    log_interval=1000,
)
dqn.test(
    env, nb_episodes=5, action_repetition=2, visualize=True, nb_max_episode_steps=100
)
dqn.save_weights("dqn_{}_weights.h5f".format("Taxi-v2"), overwrite=True)
