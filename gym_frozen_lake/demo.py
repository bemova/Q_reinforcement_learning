import gym
from gym_frozen_lake.learner import *
import argparse


def evaluate(Q, training_steps):
    reward_tracker = []
    total_rewards = 0
    state = env.reset()
    for step in range(1, training_steps):
        action = np.argmax(Q[state])
        next_state, reward, done, info = env.step(action)
        state = next_state
        total_rewards += reward
        env.render()
        if done:
            break

    reward_tracker.append(total_rewards)
    print("Average Total Return After {} Episodes: {:04.3f}".format(1, sum(reward_tracker) / 1))


if __name__ == "__main__":
    """
    The agent controls the movement of a character in a grid world.
    Some tiles of the grid are walkable, and others lead to the agent falling into the water.
    Additionally, the movement direction of the agent is uncertain and only partially depends on the chosen direction.
    The agent is rewarded for finding a walkable path to a goal tile.
    SFFF       (S: starting point, safe)
    FHFH       (F: frozen surface, safe)
    FFFH       (H: hole, fall to your doom)
    HFFG       (G: goal, where the frisbee is located)
    """
    env = gym.make("FrozenLake-v0")
    state = env.reset()
    print('current state is: {}'.format(state))

    parser = argparse.ArgumentParser(description="Run Taxi-v2 from gym library with random policy and Q learning approach")

    parser.add_argument("--approach", help="select learning approach", default="q", type=str, required=False)
    args = parser.parse_args()
    if args.approach == "q":
        print("########################start learning and create a Q matrix########################")
        Q = q_learning(env=env, alpha=0.8, gamma=0.95, epsilon=0.1, episodes=5000, training_steps=300)
        print("########################learning is done!, start testing....########################")

        print("testing for start state: {}".format(state))
        evaluate(Q, 300)
    else:
        random_policy(env=env)

