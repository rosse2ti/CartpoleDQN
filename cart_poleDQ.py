import gym
import numpy as np
from DQNAgent import DQNAgent
import time

env = gym.make("CartPole-v1", render_mode="human").env

### TRAINING


def main():
    gamma = 0.9
    epsilon = 0.95

    trials = 10
    trial_len = 200

    # updateTargetNetwork = 1000
    dqn_agent = DQNAgent(env)
    steps = []
    state_size = env.observation_space.shape[0]

    for trial in range(trials):
        done = False
        cur_state = env.reset()
        cur_state = np.reshape(cur_state[0], [1, state_size])

        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            reward = reward if not done else -10
            new_state = np.reshape(new_state, [1, state_size])
            dqn_agent.remember(cur_state, action, reward, new_state, done)
            dqn_agent.replay()  # internally iterates default (prediction) model
            dqn_agent.target_train()  # iterates target model

            cur_state = new_state
            if done:
                break

        if step >= 199:
            print("Failed to complete in trial {}".format(trial + 1))
            if step % 10 == 0:
                dqn_agent.save_model("deepQ/cart_pole/trial-{}.model".format(trial))
        else:
            print("Completed in {} trials and {} steps".format(trial, step))
            dqn_agent.save("deepQ/cart_pole/weights/dqn_cartpole_{}.h5".format(trial))


def plays():
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(env)
    agent.load("deepQ/cart_pole/weights/dqn_cartpole_4.h5")
    agent.epsilon = 0.0001
    done = False
    EPISODES = 10

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state[0], [1, state_size])
        for t in range(500):
            env.render()
            time.sleep(0.03)
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(next_state, [1, state_size])
            print(state)
            print(action)
            state = next_state
            if done:
                print(
                    "episode: {}/{}, score: {}, e: {:.2}".format(
                        e, EPISODES, t, agent.epsilon
                    )
                )
                break

    env.close()


# main()
plays()
