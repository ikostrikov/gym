import gym


env = gym.make("Reacher-v2")

episodes = 250

while True:
    env.reset()
    for i in range(episodes):
        action = env.action_space.sample()

        env.step(action)

        env.render()