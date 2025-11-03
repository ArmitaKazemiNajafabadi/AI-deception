import torch
from torch import nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical

from models.dqn_memory_buffer import MultiDQNMemory
from models.utils import combine_marl_states

MAX_SERVERS = 6
MAX_USERS = 10
MAX_EDGES = 8


def pad_sequence(seq, lens, padding):
    padded = torch.zeros(lens.size(0), padding, seq.size(-1))
    mask = torch.ones(padded.size(0), padded.size(1))

    offset = 0
    for i,len in enumerate(lens):
        st = offset
        en = offset+len

        padded[i][:len] = seq[st:en]
        mask[i][len:] = 0
        offset += len

    return padded, mask.unsqueeze(-1)

# def extract_hosts(x, servers, n_servers, users, n_users):
#     srv = x[servers]
#     srv,s_mask = pad_sequence(srv, n_servers, MAX_SERVERS)  # B x MAX_s x d

#     usr = x[users]
#     usr,u_mask = pad_sequence(usr, n_users, MAX_USERS)      # B x MAX_u x d

#     hosts = torch.cat([srv,usr], dim=1)
#     mask = torch.cat([s_mask, u_mask], dim=1)
#     return hosts, mask


class DQN(nn.Module):
    def __init__(self, input_dim,
                 hidden_dim, output_dim, lr):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.opt = Adam(self.parameters(), lr)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        self.replay_buffer = deque(maxlen=memory_capacity)

    def forward(self, x):
       # Calculate edge-level action 
       # Assume edge actions are always in groups of 8 (as they are in CAGE4)
       # Finally, compute prob of taking a global action
       # (Not an action upon a node or an edge. E.g. sleep)
       # blue_agent_4 sends in groups of 3 subnets.
       # Really makes batching tricky
         return self.fc(x)



# class InductiveCriticNetwork deleted
class DQNAgent:
    def __init__(self, agent, input_dim, hidden_dim, output_dim, lr, memory_capacity):
        self.device = device
        self.policy_net = DQN(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_net = DQN(input_dim, hidden_dim, output_dim).to(self.device)
        self.name = agent
        print(f"agent {self.name} initiation")
        try:
            raise Exception
            self.policy_net.load_state_dict(torch.load(f"dqn_policy_{agent}_net_acc.pth"))
            self.target_net.load_state_dict(torch.load(f"dqn_target_{agent}_net_acc.pth"))
            # self.policy_net.eval()
            self.target_net.eval()
            print("model loaded successfully")
        except Exception:
            traceback.print_exc()
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        # self.criterion = nn.MSELoss()
        self.criterion = nn.SmoothL1Loss()
        self.replay_buffer = deque(maxlen=memory_capacity)
        

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 3)  # 4 actions
        with torch.no_grad():
            q = self.policy_net(state.to(self.device))
            return q.argmax().item()
    """ edit """
    def store_transition(self, *transition): 
        (state, deception, reward, next_state, done) = transition
        i = f"blue_agent_{self.name}"
        agent_transition = (state[self.name], deception[self.name], reward[i], next_state[self.name], done[self.name])
        self.replay_buffer.append(agent_transition)

    def sample_batch(self, batch_size):
        return random.sample(self.replay_buffer, batch_size)

    def update(self, batch, gamma):
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.cat(states).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target = -rewards + gamma * next_q * (1-dones) 

        loss = self.criterion(q_values, target)
        # loss = nn.SmoothL1Loss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print(f"loss is ({loss})")
        return loss.item()
        

class DQNAgent():
    '''
    Class to manage agents' memories and learning (when training)
    When training is complete, uses the InductiveActorNetwork to decide
    which action to take
    '''
    def __init__(self, in_dim, gamma,  bs, 
                 a_kwargs=dict(), c_kwargs=dict(), training=True):

        self.actor = DQN(in_dim, **a_kwargs) """ modify """
        self.memory = MultiDQNMemory(bs, agents=5)

        self.args = (in_dim,)
        self.kwargs = dict(
            gamma=gamma, bs=bs,
            a_kwargs=a_kwargs, c_kwargs=c_kwargs, training=training
        )

        # PPO Hyperparams
        self.gamma = gamma
        self.bs = bs

        self.training = training
        self.deterministic = False
        self.mse = nn.MSELoss() """Thee other? wwhy here we need loss aslan? Renamed to loss_fn; used to compute TD error (MSE loss between Q-values and targets) """

    # Required by CAGE but not utilized
    def end_episode(self):
        pass

    # Required by CAGE but not utilized
    def set_initial_values(self, action_space, observation):
        pass

    def train(self):
        '''
        Set modules to training mode
        '''
        self.training = True 
        self.actor.train()  """would my actor be okay?"""

    def eval(self):
        '''
        Set modules to eval mode 
        '''
        self.training = False
        self.actor.eval()

    def _zero_grad(self):
        '''
        Reset opt
        '''
        self.actor.opt.zero_grad()

    def _step(self):
        '''
        Call opt autograd
        '''
        self.actor.opt.step()

    def set_deterministic(self, val):
        self.deterministic = val

    def set_mems(self, mems):
        self.memory.mems = mems

    def save(self, outf='saved_models/dqn.pt'):
        me = (self.args, self.kwargs)

        torch.save({
            'actor': self.actor.state_dict(),
            'agent': me
        }, outf)


    """ function modified with some remaining shak in the end"""
    @torch.no_grad()
    def get_action(self, obs, *args):
        '''
        Sample an action from the actor's distribution
        given the current state. 

        If eval(), only returns the action 
        If train() returns action, value, and log prob 
        '''
        state,is_blocked = obs
        if is_blocked:
            return None

        q_values  = self.actor(*state)

        # I don't know why this would ever be called
        # during training, but just in case, putting the
        # logic block outside the training check
        if self.deterministic or torch.rand(1).item() > epsilon:
            action = q_values.argmax().item()
        else:
            action = torch.randint(0, q_values.shape[-1], (1,)).item()

        # if not self.training:
        #     return action.item()

        return action

    def remember(self, idx, s, a, r, ns, t):
        '''
        Save an observation to the agent's memory buffer
        '''
        self.memory.remember(idx, s, a, r, ns, t)

    def learn(self, verbose=False):
        '''        
        This runs the PPO update algorithm on memories stored in self.memory 
        Assumes that an external process is adding memories to the buffer
        '''
        if len(self.memory) < self.batch_size:
            return  # Not enough data yet
            
        s, a, r, ns, t, batches = self.memory.get_batches()
    
        # Preprocess state batches
        states = combine_marl_states(s)
        next_states = combine_marl_states(ns)


        # Q-values for current states
        q_values = self.actor(*states)  # shape: [batch, num_actions]
        q_values = q_values.gather(1, torch.tensor(a).unsqueeze(1))  # Q(s,a)

        # Q-values for next states (target net)
        with torch.no_grad():
            next_q_values = target_net(*next_states)
            max_next_q = next_q_values.max(dim=1)[0]  # max_a Q(s', a)
            targets = torch.tensor(-r) + self.gamma * max_next_q * (1 - torch.tensor(t, dtype=torch.float))
    
        # Compute loss and update
        loss = self.loss_fn(q_values.squeeze(), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if verbose:
            print(f"DQN loss: {loss.item():.4f}")
    
        return loss.item()
        
       
        # Optimize for clipped advantage for each minibatch 
        # for b_idx,b in enumerate(batches):
        #     b = b.tolist()
        #     new_probs = []

        #     # Combine graphs from minibatches so GNN is called once
        #     s_ = [s[idx] for idx in b]
        #     a_ = [a[idx] for idx in b]
        #     batched_states = combine_marl_states(s_)
        #     self._zero_grad()
        #     # Print loss for each minibatch if verbose 
        #     # (aggregate loss is printed regardless)
        #     # Print avg loss across minibatches


        # After we have sampled our minibatches e times, clear the memory buffer
        # self.memory.clear()
        # return total_loss.item()


def load(in_f):
    '''
    Loads model checkpoint file 
    '''
    data = torch.load(in_f)
    args,kwargs = data['agent']

    agent = DQNAgent(*args, **kwargs)
    agent.actor.load_state_dict(data['actor'])
    agent.eval()
    return agent

