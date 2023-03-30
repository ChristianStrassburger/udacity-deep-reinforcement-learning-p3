from ddpg_hyperparameters import DDPGHyperparameters
from typing import List
import numpy as np

class DDPGGridsearch():

    def __init__(self, lr_actor_params = [1.5e-3], lr_critic_params = [1.5e-3], 
                 gamma_params = [0.99], buffer_size_params = [int(1e5)], batch_size_params = [128], 
                 weight_decay_params = [0], tau_params = [0.003], random_seed_params = [1],
                 fc1_units_a_params = [300], fc2_units_a_params = [200], fc3_units_a_params = [100],
                 fc1_units_c_params = [200], fc2_units_c_params = [100]):
        """Initialize a DDPGGridsearch object.
        
        Params
        ======
            lr_actor_params (array): actor learning rate parameters
            lr_critic_params (array): critic learning rate parameters
            gamma_params (array): discount factor parameters
            buffer_size_params (array): replay buffer size parameters
            batch_size_params (array): batch size parameters
            weight_decay_params (array): L2 weight decay parameters
            tau_params (array): for soft update of target parameters
            random_seed_params (array): random_seed parameters
            fc1_units_a_params (array): number of nodes in first hidden layer parameters (actor)
            fc2_units_a_params (array): number of nodes in second hidden layer parameters (actor)
            fc3_units_a_params (array): number of nodes in third hidden layer parameters (actor)
            fc1_units_c_params (array): number of nodes in first hidden layer parameters (critic)
            fc2_units_c_params (array): number of nodes in second hidden layer parameters (critic)
        """
        self.lr_actor_params = lr_actor_params
        self.lr_critic_params = lr_critic_params
        self.gamma_params = gamma_params
        self.buffer_size_params = buffer_size_params
        self.batch_size_params = batch_size_params
        self.weight_decay_params = weight_decay_params
        self.tau_params = tau_params
        self.random_seed_params = random_seed_params
        self.fc1_units_a_params = fc1_units_a_params
        self.fc2_units_a_params = fc2_units_a_params
        self.fc3_units_a_params = fc3_units_a_params
        self.fc1_units_c_params = fc1_units_c_params
        self.fc2_units_c_params = fc2_units_c_params

    def create_gridsearch_params(self) -> List:
        """Returns an array with DDPGHyperparameters objects."""
        param_list = [(lr_a, ) for lr_a in self.lr_actor_params]
        param_list = [(*m, lr_c) for m in param_list for lr_c in self.lr_critic_params]
        param_list = [(*m, g) for m in param_list for g in self.gamma_params]
        param_list = [(*m, d) for m in param_list for d in self.weight_decay_params]
        param_list = [(*m, bf) for m in param_list for bf in self.buffer_size_params]
        param_list = [(*m, bs) for m in param_list for bs in self.batch_size_params]
        param_list = [(*m, t) for m in param_list for t in self.tau_params]
        param_list = [(*m, rs) for m in param_list for rs in self.random_seed_params]
        param_list = [(*m, fc1) for m in param_list for fc1 in self.fc1_units_a_params]
        param_list = [(*m, fc2) for m in param_list for fc2 in self.fc2_units_a_params]
        param_list = [(*m, fc3) for m in param_list for fc3 in self.fc3_units_a_params]
        param_list = [(*m, fc1) for m in param_list for fc1 in self.fc1_units_c_params]
        param_list = [(*m, fc2) for m in param_list for fc2 in self.fc2_units_c_params]

        param_list = [DDPGHyperparameters(lr_actor=p[0], lr_critic=p[1], gamma=p[2], 
                                          weight_decay=p[3], buffer_size=p[4], batch_size=p[5], 
                                          tau=p[6], random_seed=p[7], fc1_units_a=p[8], fc2_units_a=p[9],
                                          fc3_units_a=p[10], fc1_units_c=p[11], 
                                          fc2_units_c=p[12]) for p in param_list]

        return param_list
    
    def __str__(self) -> str:
        """Returns a DDPGGridsearch object as a string."""
        return f"""lr_actor: {self.lr_actor_params}
lr critic: {self.lr_critic_params}
gamma: {self.gamma_params}
buffer size: {self.buffer_size_params}
batch size: {self.batch_size_params}
weight decay: {self.weight_decay_params}
random seed: {self.random_seed_params}
fc1 units actor: {self.fc1_units_a_params}
fc2 units actor: {self.fc2_units_a_params}
fc3 units actor: {self.fc3_units_a_params}
fc1 units critic: {self.fc1_units_c_params}
fc2 units critic: {self.fc2_units_c_params}"""