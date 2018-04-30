import gym
from gym_taxi_2.learner import *
import argparse

if __name__ == "__main__":
    """
    You receive +20 points for a successful dropoff,
    and lose 1 point for every timestep it takes.
    There is also a 10 point penalty for illegal pick-up and drop-off actions.
    actions: down (0), up (1), right (2), left (3), pick-up (4), and drop-off (5)
    """
    env = gym.make("Taxi-v2")
    state = env.reset()
    print('current state is: {}'.format(state))
    parser = argparse.ArgumentParser(description="Run Taxi-v2 from gym library with random policy and Q learning approach")

    parser.add_argument("--approach", help="select learning approach", default="q", type=str, required=False)
    args = parser.parse_args()

    if args.approach == "q":
        print("########################start learning and create a Q matrix########################")
        Q = q_learning(env=env)
        print("########################learning is done!, start testing....########################")
        state = env.reset()
        print("testing for start state: {}".format(state))
        done = None
        while done != True:
            # Select the action with the highest Q Value
            action = np.argmax(Q[state])
            state, reward, done, info = env.step(action)
            env.render()
    else:
        random_policy(env=env)

