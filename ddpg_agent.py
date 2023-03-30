import numpy as np
import random
from actor import Actor
from critic import Critic
from ounoise import OUNoise
from replay_buffer import ReplayBuffer
import torch
import torch.nn.functional as F
import torch.optim as optim

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, device, hyperparameters, state_size, action_size, random_seed=0, num_agents=2):
        """Initialize an Agent object.
        
        Params
        ======
            device (str): device name - 'cpu' or 'cuda:0'
            hyperparameters (DDPGHyperparameters): hyperparameters
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            num_agents (int): number of agetns
        """
        self.device = device
        self.hyperparameters = hyperparameters
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = num_agents

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed,
                                 fc1_units= hyperparameters.fc1_units_a,
                                 fc2_units= hyperparameters.fc2_units_a,
                                 fc3_units= hyperparameters.fc3_units_a).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed,
                                 fc1_units= hyperparameters.fc1_units_a,
                                 fc2_units= hyperparameters.fc2_units_a,
                                 fc3_units= hyperparameters.fc3_units_a).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=hyperparameters.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed,
                                   fcs1_units= hyperparameters.fc1_units_c,
                                   fc2_units= hyperparameters.fc2_units_c).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed,
                                    fcs1_units= hyperparameters.fc1_units_c,
                                    fc2_units= hyperparameters.fc2_units_c).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), 
                                           lr=hyperparameters.lr_critic, weight_decay=hyperparameters.weight_decay)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(device=self.device, action_size=self.action_size, 
                                   buffer_size=self.hyperparameters.buffer_size, 
                                   batch_size=self.hyperparameters.batch_size, seed = random_seed)
        
        self.soft_update(self.critic_local, self.critic_target, 1.0)
        self.soft_update(self.actor_local, self.actor_target, 1.0) 
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for idx in range(self.num_agents):
            self.memory.add(states[idx], actions[idx], rewards[idx], next_states[idx], dones[idx])

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.hyperparameters.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.hyperparameters.gamma)

    def act(self, state, add_noise=False):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()

        action = np.clip(action, -1, 1)
        return action

    def reset(self):
        """Reset the OUNoise object."""
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        
        #print(f"actions_next: {actions_next.shape}")
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.hyperparameters.tau)
        self.soft_update(self.actor_local, self.actor_target, self.hyperparameters.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



