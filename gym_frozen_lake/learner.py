import numpy as np


def random_policy(env):
    counter = 0
    total_rewards = 0
    reward = None
    rewardTracker = []
    while reward != 1:
        env.render()
        state, reward, done, info = env.step(env.action_space.sample())
        total_rewards += reward
        if done:
            rewardTracker.append(total_rewards)
            env.reset()
            counter += 1
    print("Solved in {} Steps with a average return of {}".format(counter, sum(rewardTracker) / len(rewardTracker)))


def epsilon_greedy(env, epsilon, Q, state, episode):
    n_actions = env.action_space.n
    if np.random.rand() > epsilon:
        # adding a noise to the best action from Q
        action = np.argmax(Q[state, :] + np.random.randn(1, n_actions) / (episode / n_actions))
    else:
        action = env.action_space.sample()
        # reduce the epsilon number
        epsilon -= 10 ** -5
    return action, epsilon


def q_learning(env, alpha, gamma, epsilon, episodes, training_steps):
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    reward_tracker = []

    for episode in range(1, episodes + 1):

        total_rewards = 0
        state = env.reset()

        for step in range(1, training_steps):
            action, epsilon = epsilon_greedy(env, epsilon, Q, state, episode)
            next_state, reward, done, info = env.step(action)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state
            total_rewards += reward

        reward_tracker.append(total_rewards)

        if episode % (episodes * .10) == 0 and episode != 0:
            print('Epsilon {:04.3f}  Episode {}'.format(epsilon, episode))
            print("Average Total Return: {}".format(sum(reward_tracker) / episode))

        if (sum(reward_tracker[episode - 100:episode]) / 100.0) > .78:
            print('Solved after {} episodes with average return of {}'
                  .format(episode - 100, sum(reward_tracker[episode - 100:episode]) / 100.0))
            return Q
    return Q