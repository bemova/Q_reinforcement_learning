import numpy as np


def random_policy(env):
    counter = 0
    total_rewards = 0
    reward = None
    while reward != 20:
        env.render()
        state, reward, done, info = env.step(env.action_space.sample())
        counter += 1
        total_rewards += reward
    print("Solved in {} Steps with a total reward of {}".format(counter, total_rewards))


# random_policy() # result : Solved in 3768 Steps with a total reward of -14556

def q_learning(env, episodes=1000, reward_decay=0.9, alpha=0.618):
    reward_tracker = []
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros([n_states, n_actions])
    total_rewards = 0
    for episode in range(1, episodes + 1):
        done = False
        total_rewards, reward = 0, 0
        state = env.reset()
        while not done:
            action = np.argmax(Q[state])
            next_state, reward, done, info = env.step(action)
            Q[state, action] += alpha * (reward + (reward_decay * np.max(Q[next_state])) - Q[state, action])
            total_rewards += reward
            state = next_state
        reward_tracker.append(total_rewards)

        if episode % 100 == 0:
            print('Episode: %5d,\t\tReward: %4d,\t\tTotal Average Reward:   %5.5f '
                   % (episode, total_rewards, sum(reward_tracker) / len(reward_tracker)))
    return Q

# Q = q_learning()
# state = env.reset()
# done = None
# print(Q)
# while done != True:
#     # We simply take the action with the highest Q Value
#     action = np.argmax(Q[state])
#     state, reward, done, info = env.step(action)
#     env.render()
