import torch
import numpy as np
from env import TRex
from agent import Agent

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = TRex()
    agent = Agent(env, device)

    for _ in range(10000):
        state = env.reset()
        agent.memory_reset()

        t = 0
        done = False
        while not done:
            t += 1
            
            action = agent.select_action(state)

            next_state, reward, done = env.step(action)

            reward = torch.FloatTensor(np.array([reward])).to(device)
            done = torch.FloatTensor(np.array([done], dtype=np.float)).to(device)

            agent.rewards.append(reward)
            agent.dones.append(done)

            state = next_state

            if t >= 100 or done:
                print("Train")
                agent.train_model(state)
                t = 0

        print("Reward", reward)
        
        