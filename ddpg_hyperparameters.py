class DDPGHyperparameters():

    def __init__(self,  lr_actor, lr_critic, gamma = 0.99, buffer_size = int(1e6), batch_size = 128, 
                tau = 1e-3, weight_decay = 0, random_seed = 0, fc1_units_a = 300, fc2_units_a = 200,
                fc3_units_a = 100, fc1_units_c = 200, fc2_units_c = 100):
        """Initialize a DDPGHyperparameters object.
        
        Params
        ======
            lr_actor (float): learning rate actor
            lr_critic (float): learning rate critic
            gamma (float): discount factor
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            tau (float): for soft update of target parameters
            weight_decay (float): L2 weight decay
            random_seed (int): random seed
            fc1_units_a (int): number of nodes in first hidden layer (actor)
            fc2_units_a (int): number of nodes in second hidden layer (actor)
            fc3_units_a (int): number of nodes in third hidden layer (actor)
            fc1_units_c (int): number of nodes in first hidden layer (critic)
            fc2_units_c (int): number of nodes in second hidden layer (critic)
        """
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.tau = tau
        self.weight_decay = weight_decay
        self.random_seed = random_seed
        self.fc1_units_a = fc1_units_a
        self.fc2_units_a = fc2_units_a
        self.fc3_units_a = fc3_units_a
        self.fc1_units_c = fc1_units_c
        self.fc2_units_c = fc2_units_c

    def __str__(self) -> str:
        """Returns a DDPGHyperparameters object as a string."""
        return f"""lr_actor: {self.lr_actor}
lr_critic: {self.lr_critic}
gamma: {self.gamma}
buffer_size: {self.buffer_size}
batch_size: {self.batch_size}
tau: {self.tau}
weight_decay: {self.weight_decay}
random_seed: {self.random_seed}
fc1_units actor: {self.fc1_units_a}
fc2_units actor: {self.fc2_units_a}
fc3_units actor: {self.fc3_units_a}
fc1_units critic: {self.fc1_units_c}
fc2_units critic: {self.fc2_units_c}"""