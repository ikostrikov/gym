import gym


env = gym.make("Ant-v3")

episodes = 250

while True:
    env.reset()
    for i in range(episodes):
        action = env.action_space.sample()

        env.step(action)

        env.render()