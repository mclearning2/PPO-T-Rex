import torch
import torch.optim as optim
import numpy as np
from model import Net

class Agent:
    def __init__(self, env, device):
        self.env = env
        self.device = device

        self.batch_size = 32
        self.epsilon = 0.2
        self.n_epoch = 4
        self.gamma = 0.99
        self.tau = 0.96
        
        self.model = Net(env.state_size, env.action_size).to(device)
        self.optim = optim.Adam(self.model.parameters())
    
    def memory_reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.log_probs = []

    def select_action(self, state):

        state = torch.FloatTensor(state).to(self.device).unsqueeze(0) / 255.
        
        policy, value = self.model(state)
        
        policy, value = policy[0], value[0]

        probablity = policy.cpu().detach().numpy()
        action = np.random.choice(self.env.action_size, 1, p=probablity)[0]
        log_probs = torch.log(policy[action])

        self.states.append(state)
        self.actions.append(torch.LongTensor([action]))
        self.values.append(value)
        self.log_probs.append(log_probs.unsqueeze(0))

        return action

    def compute_gae(self, last_value):
        values = self.values + [last_value[0]]
        gae = 0
        returns = list()

        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step]  \
                  + self.gamma * values[step + 1] * (1 - self.dones[step]) \
                  - values[step]
            gae = delta + self.gamma * self.tau * (1 - self.dones[step]) * gae
            returns.insert(0, gae + values[step])

        return returns

    def ppo_iter(self, states, actions, log_probs, returns, advantage):
        # states.size(0) = rollout_len * n_workers
        memory_size = states.size(0)        

        for _ in range(memory_size // self.batch_size):
            random_indices = np.random.randint(0, memory_size, self.batch_size)
            yield states[random_indices], \
                  actions[random_indices], \
                  log_probs[random_indices], \
                  returns[random_indices], \
                  advantage[random_indices]

    def ppo_update(self, states, actions, log_probs, returns, advantage):
        epsilon = self.epsilon        

        for _ in range(self.n_epoch):
            for state, action, old_log_probs, return_, adv in \
                self.ppo_iter(states, actions, log_probs, returns, advantage):

                policy, value = self.model(state)

                # Actor Loss
                # ============================================================
                new_log_probs = policy[0][action]
                ratio = (new_log_probs - old_log_probs).exp()                
                
                surr_loss = ratio * adv
                clipped_surr_loss = torch.clamp(ratio, 1.0-epsilon, 1.0+epsilon) * adv         

                actor_loss = - torch.min(surr_loss, clipped_surr_loss).mean()
                # ============================================================

                # Critic Loss
                #TODO: clip value by epsilon, clip gradient by norm
                # ============================================================
                critic_loss = (return_ - value).pow(2).mean()
                # ============================================================

                loss = actor_loss + 0.5 * critic_loss

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()


    def train_model(self, last_state):
        last_state = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
        _, last_value = self.model(last_state)

        returns = self.compute_gae(last_value)

        returns = torch.cat(returns).detach() 
        log_probs = torch.cat(self.log_probs).detach() 
        values = torch.cat(self.values).detach()
        states = torch.cat(self.states) 
        actions = torch.cat(self.actions)

        advantage = returns - values

        self.ppo_update(states, actions, log_probs, returns, advantage)