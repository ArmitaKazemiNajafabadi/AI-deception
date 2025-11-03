from copy import deepcopy
# 
import numpy as np
import torch
from random import random, choice, randint
import torch.nn as nn
import torch.nn.functional as F
import hashlib



from CybORG.env import CybORG
from CybORG.Agents.Wrappers.EnterpriseMAE import EnterpriseMAE
from CybORG.Simulator.Actions.Action import Sleep
from CybORG.Shared.Enums import TernaryEnum

from wrapper.observation_graph import ObservationGraph
from wrapper.globals import *

# class DQN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             # nn.Linear(hidden_dim, 32),
#             # nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim)
#         )

#     def forward(self, x):
#         return self.fc(x)


class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, output_dim)
        )

    def forward(self, x):
        return self.fc(x)        

class STGraphWrapper(EnterpriseMAE):
    def __init__(self, env: CybORG, *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        self.graphs = dict()
        self.env = env
        self.agent_names = [f'blue_agent_{i}' for i in range(5)]
        self.ts = 0
        
        self.msg = {
            a:np.zeros(8)
            for a in self.agent_names
        }

        self.submission = None
        ### ADDED TO LOAD MANIPULATION AGNET ####
        self.SWITCH_ACTIONS = {                 # "0" corresponds to "no switch"
            1: TernaryEnum.TRUE,
            2: TernaryEnum.FALSE,
            3: TernaryEnum.UNKNOWN
        }
        # Define and load model for EVAL
        self.dqn_model = []
        self.deceptions = np.zeros(5, int)
        self.last_decept = [False, False, False, False, False]
        self.manp_dict_obs = {}
        self.gen_obs = {}
        self.manp_obs = {}
        for i in range(5):
            agent = f'blue_agent_{i}'
            self.manp_dict_obs[agent] = self.env.environment_controller.get_last_observation(agent).data

        # print(f'self.manp_dict_obs = {self.manp_dict_obs}')
        for i in range(5):
            # self.dqn_model.append(DQN(input_dim=36, hidden_dim=64, output_dim=4)) #if input_dim > 200 set hidden_dim 256
            # self.dqn_model[i].load_state_dict(torch.load(f'dqn_policy_{i}_net_acc.pth'))
            
            # if i < 4:
            #     self.dqn_model.append(DQN(input_dim=258, hidden_dim=256, output_dim=7)) #if input_dim > 200 set hidden_dim 256
            # else: 
            #     self.dqn_model.append(DQN(input_dim=494, hidden_dim=512, output_dim=7)) #if input_dim > 200 set hidden_dim 256
            if i < 4:
                self.dqn_model.append(DQN(input_dim=130, hidden_dim_1=256, output_dim=7)) #if input_dim > 200 set hidden_dim 256
            else: 
                self.dqn_model.append(DQN(input_dim=248, hidden_dim_1=512, output_dim=7)) #if input_dim > 200 set hidden_dim 256

            # self.dqn_model[i].load_state_dict(torch.load(f'dqn_Turbo_policy_{i}_net_Jul14.pth'))
            # self.dqn_model[i].load_state_dict(torch.load(f'dqn_Single_Turbo_Greedy_policy_{i}_net_Jul20.pth'))
            self.dqn_model[i].load_state_dict(torch.load(f'dqn_Single_Turbo_policy_{i}_net_Jul22.pth'))
            
            self.dqn_model[i].eval() #  sets the model to inference mode

        self.M = 500
        self.FIRST_TIME = True
        print("env initialized")
        # saving list
        self.action_adv_list = np.zeros((5, 100, 500)) - 1
        self.action_def_list = np.empty((5, 100, 500), str) 
        self.observation_list = []
        self.reward_list = np.zeros((100, 500))
        self.mssg_list = []
        self.mssg_dis_list = []
        self.budget = np.zeros((100, 500))
        self.step_count = 0
        self.reset_count = 0

        print("in init: " + str(self.reset_count))

        #### END ####

    def action_translator(self, agent_name, a_id):
        '''
        Translates output of PPO model to an action for the CybORG env. 
        Model provides output as
        Node-actions, edge-actions, global-actions, per-subnet.
        '''
        session = 0 # Seems the same every time?

        if a_id is None:
            return Monitor(session, agent_name)

        agent_id = int(agent_name[-1])
        which_subnet = MY_SUBNETS[agent_id][a_id // MAX_ACTIONS]
        a_id %= MAX_ACTIONS

        # Node action
        if a_id < N_NODE_ACTIONS*MAX_HOSTS:
            a = NODE_ACTIONS[a_id // MAX_HOSTS]
            target = a_id % MAX_HOSTS

            if target > 5:
                target = f'{which_subnet}_user_host_{target-6}'
            else:
                target = f'{which_subnet}_server_host_{target}'

            return a(session=session, agent=agent_name, hostname=target)

        # Edge action
        elif (a_id := a_id - (N_NODE_ACTIONS*MAX_HOSTS)) < (N_EDGE_ACTIONS*POSSIBLE_NEIGHBORS):
            a = EDGE_ACTIONS[a_id // POSSIBLE_NEIGHBORS]
            target = [r for r in ROUTERS if which_subnet not in r][a_id % POSSIBLE_NEIGHBORS]

            return a(session, agent_name, target.replace('_router',''), which_subnet)

        # Global action (only one)
        else:
            return Monitor(session, agent_name)


    # TRAINING
    def modified_step(self, deceptions, observations):
        # print(deception)
        # print("observation")
        # print(observations)
        actions = {
            agent_name: agent.get_action(
                observations[agent_name], self.get_action_space(agent_name)
            )
            for agent_name, agent in self.submission.AGENTS.items()
            if agent_name in self.env.agents
        }
        # print("actionS: ")
        # print(actions)
        action = {
            k:self.action_translator(k,v)
            for k,v in actions.items()
        }

        # print(f"in modified_step: actions{actions}")
        observation, reward, term, trunc, info = super().step(
        action_dict=action, messages=self.msg
    )
        self.gen_obs = deepcopy(observation)
        ## MODIFY REQUIRED
        dict_obs_all = {}
        for i in range(5):
            agent = f'blue_agent_{i}'
            # print(agent)
                    # Get the raw observation dictionary for this agent
            dict_obs = self.env.environment_controller.get_last_observation(agent).data
            
            current_success = dict_obs.get('success', TernaryEnum.UNKNOWN)
            # Do not switch if still IN_PROGRESS
            
        
            if (deceptions[i] in self.SWITCH_ACTIONS) and (self.M>0) and (current_success != TernaryEnum.IN_PROGRESS) and 'message' in dict_obs:
                dict_obs['success'] = self.SWITCH_ACTIONS[deceptions[i]]
                self.M -=1
            if (deceptions[i] > 3) and (self.M>0):
                idx = 27 + 1
                if deceptions[i] == 4:
                    # print(f"prevent bad process detection :{i}")
                    if i < 4:
                        observation[agent][idx:idx+16] = 0
                    elif i == 4:
                        observation[agent][idx:idx+16] = 0
                        observation[agent][idx+59:idx+59+16] = 0
                        observation[agent][idx+118:idx+118+16] = 0      
                    self.M -=1
                elif deceptions[i] == 5:
                    # print(f"prevent both bad process and bad event")
                    if i < 4:
                        observation[agent][idx:idx+16+16] = 0
                    elif i == 4:
                        observation[agent][idx:idx+16+16] = 0
                        observation[agent][idx+59:idx+59+16+16] = 0
                        observation[agent][idx+118:idx+118+16+16] = 0
                    self.M -=2 # IT IS POSSIBLE THAT THE "M" IS ALREADY 1 and CAN NOT PERFORM THIS ACTION.
    
             # generate dict_obs_all from dict_obs
            dict_obs_all[agent] = dict_obs.copy()
        
        # Tell ObservationGraph what happened and update
        graph_obs = dict()
        for i in range(5):
            agent = f'blue_agent_{i}'


            # Get the raw observation dictionary for this agent
            # print(dict_obs_all)
            dict_obs = dict_obs_all[agent].copy()
        
            
            msg = dict_obs.pop('message')
            msg = np.stack(msg, axis=0)
            # print(f"this is agent {i} poped message from dict obss {msg.astype(int)} from obs {observation[agent][-32:]}")
            # Indicates if msg was recieved or comms are blocked
            # This way we differentiate between feature for 0 and unknown
            recieved_msg = msg[:, -1:]
            if i != 4:
                # Repeat agent 4's 'is_recieved' message across 2 more subnets
                recieved_msg = np.concatenate([recieved_msg, np.zeros((2,1))], axis=0)
                recieved_msg[-2:] = recieved_msg[-3]

                # Pull out messages for 'was_scanned' and 'was_comprimised'
                msg_small = msg[:-1, :2]
                msg_big = msg[-1, :6].reshape(3,2)
                msg = np.concatenate([msg_small, msg_big], axis=0)
            else:
                msg = msg[:, :2]

            msg = np.concatenate([msg, recieved_msg], axis=1)
            if (i != 4) and (deceptions[i] == 6) and (self.M > 0):
                  # print(f"deception on mssg happened i < 4 original msg {msg}")
                  msg = np.array([[0., 0., 1.], [0., 0., 1.],  [0., 0., 1.],  [0., 0., 1.],  [0., 0., 1.],  [0., 0., 1.]])
                  observation[agent][-32:] = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1], dtype='int')
                  # print(f"print observation agent{agent}: {(dict_obs_all[agent].get('message'))}")
                  dict_obs_all[agent].update({"message": [np.array([False, False, False, False, False, False, False,  True]), np.array([False, False, False, False, False, False, False,  True]), np.array([False, False, False, False, False, False, False,  True]), np.array([False, False, False, False, False, False, False,  True])]})
                  self.M -= 1
            elif (i == 4) and (deceptions[i] == 6) and (self.M > 0):
                    # print("deception on mssg happened i = 4")
                    msg = np.array([[False, False,  True], [False, False,  True], [False, False,  True], [False, False,  True]])
                    # print(f"print observation agent{agent}: {(observation[agent][-32:])}")
                    # print(f"print observation agent{agent}: {(dict_obs_all[agent].get('message'))}")
                    observation[agent][-32:] = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1], dtype='int')
                    dict_obs_all[agent].update({"message": [np.array([False, False, False, False, False, False, False,  True]), np.array([False, False, False, False, False, False, False,  True]), np.array([False, False, False, False, False, False, False,  True]), np.array([ False, False, False, False, False, False, False,  True])]})
                    self.M -= 1
            # Update the graph based on the raw dictionary 
            self.manp_obs = deepcopy(observation)
            o = observation[agent]
            g = self.graphs[agent]
            g.parse_observation(dict_obs)

            # Pull node features from tabular observation, and also update 
            # graph subnet connectivity edges. 
            tab_x,phase,new_msg = self._parse_tabular(o, g) 
            # tab_x,phase,new_msg = self._parse_tabular(o, g, deceptions) 
            
            self.msg[agent] = new_msg
            # print(f"new message agent {} is {self.msg[agent]}")

            # Combine node features from graph source, and tabular source
            x,ei,masks = g.get_state(MY_SUBNETS[i])
            x = self._combine_data(x, tab_x)

            # Mask/pack into conviniently sized tensors for the GNN models 
            obs = self._to_obs(x,ei,masks,phase,msg,new_msg)

            # During training, we need to know if the agent is still mid-action
            # If so, we don't bother calculating an action next turn 
            is_blocked = dict_obs['success'] == TernaryEnum.IN_PROGRESS
            graph_obs[agent] = (obs, is_blocked)

        self.ts += 1
        self.last_obs = graph_obs
        # print("function done \n")
        return graph_obs, dict_obs_all, reward, term, trunc, info, actions, self.gen_obs, self.manp_obs

    
    ##  EVALUATION
    def modify_required(self, observation, reward, term, trunc, info):
        dict_obs_all = {}
        for i in range(5):
            agent = f'blue_agent_{i}'
            # print(agent)
                    # Get the raw observation dictionary for this agent
            dict_obs = self.env.environment_controller.get_last_observation(agent).data
            # print(f"dict_obs for {agent}: {dict_obs}")
            current_success = dict_obs.get('success', TernaryEnum.UNKNOWN)
            # Do not switch if still IN_PROGRESS
            """""" # for the dqn edit
            # obs_tensor = encode_dict_obs(dict_obs)
            obs_tensor = self.encode_dict_obs_expnd_inpt(dict_obs,  self.gen_obs[f'blue_agent_{i}'])

            with torch.no_grad():
                q_values = self.dqn_model[i](obs_tensor)
                action = torch.argmax(q_values).item()
                # action = randint(0, 6)
                # if i == 0:
                    # action = 0
                # action = 0
                # print(action)
                self.deceptions[i] = action 

            

            # action = randint(0, 3)
            # print(action)
            # print(f"action for agent {i} is: {action}")
            
            if (action in self.SWITCH_ACTIONS) and (self.M > 0) and (current_success != TernaryEnum.IN_PROGRESS) and ('message' in dict_obs):
                dict_obs['success'] = self.SWITCH_ACTIONS[action]
                self.M -= 1
                # print(f"action switch for agent {i} happened")
            
            if (action > 3) and (self.M>0):
                idx = 27 + 1
                if action == 4:
                    # print(f"prevent bad process detection :{i}")
                    if i < 4:
                        observation[agent][idx:idx+16] = 0
                    elif i == 4:
                        observation[agent][idx:idx+16] = 0
                        observation[agent][idx+59:idx+59+16] = 0
                        observation[agent][idx+118:idx+118+16] = 0      
                    self.M -=1
                elif action == 5:
                    # print(f"prevent both bad process and bad event")
                    if i < 4:
                        observation[agent][idx:idx+16+16] = 0
                    elif i == 4:
                        observation[agent][idx:idx+16+16] = 0
                        observation[agent][idx+59:idx+59+16+16] = 0
                        observation[agent][idx+118:idx+118+16+16] = 0
                    self.M -=2 # IT IS POSSIBLE THAT THE "M" IS ALREADY 1 and CAN NOT PERFORM THIS ACTION.
                # elif action == 6:
                #     if i < 4:
                        
                #     elif i == 4:   
                        
            dict_obs_all[agent] = dict_obs.copy()
        
            # print(f"this should be the dict_obs_all: {dict_obs_all}")
        return observation, reward, term, trunc, info, dict_obs_all

    #Evaluation
    def step(self, action):
        # print(self.M)
        # if self.FIRST_TIME:
        #     from submission import Submission
        #     self.FIRST_TIME = False
        #     self.submission = Submission()
        '''
        Take an action, execute it, and update internal approximation of the
        environment based on new observation.

        Args: 
            action: dict of {agent_id (int) : action_id (int)}
        '''
        # Convert from model out to Action objects
        action = {
            k:self.action_translator(k,v)
            for k,v in action.items()
        }
        # print(f"message {self.msg}")
        # Gets the info from the tabular wrapper (4 dims per host, in order)
        observation, reward, term, trunc, info = super().step(
            action_dict=action, messages=self.msg
        )
        self.gen_obs = deepcopy(observation)
        observation, reward, term, trunc, info, dict_obs_all = self.modify_required(observation, reward, term, trunc, info)
        # print(f"term{term}")
        # print(f"trun{trunc}")
        # print(f"info{info}")
        # print(f"dict_{dict_obs_all}")
        # print(f"Observatoin: {observation}")
        # Tell ObservationGraph what happened and update
        graph_obs = dict()
        for i in range(5):
            agent = f'blue_agent_{i}'


            # Get the raw observation dictionary for this agent
            # print(dict_obs_all)
            dict_obs = dict_obs_all[agent].copy()
  
            
            msg = dict_obs.pop('message')
            msg = np.stack(msg, axis=0)

            # Indicates if msg was recieved or comms are blocked
            # This way we differentiate between feature for 0 and unknown
            recieved_msg = msg[:, -1:]
            if i != 4:
                # Repeat agent 4's 'is_recieved' message across 2 more subnets
                recieved_msg = np.concatenate([recieved_msg, np.zeros((2,1))], axis=0)
                recieved_msg[-2:] = recieved_msg[-3]

                # Pull out messages for 'was_scanned' and 'was_comprimised'
                msg_small = msg[:-1, :2]
                msg_big = msg[-1, :6].reshape(3,2)
                msg = np.concatenate([msg_small, msg_big], axis=0)
            else:
                msg = msg[:, :2]



            # print(f"received msg: {recieved_msg}, msg for agent {i} is: {msg}")
            msg = np.concatenate([msg, recieved_msg], axis=1)
            if (i != 4) and (self.deceptions[i] == 6) and (self.M > 0):
                  # print(f"deception on mssg happened i < 4 original msg {msg}")
                  msg = np.array([[0., 0., 1.], [0., 0., 1.],  [0., 0., 1.],  [0., 0., 1.],  [0., 0., 1.],  [0., 0., 1.]])
                  observation[agent][-32:] = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1], dtype='int')
                  # print(f"print observation agent{agent}: {(dict_obs_all[agent].get('message'))}")
                  dict_obs_all[agent].update({"message": [np.array([False, False, False, False, False, False, False,  True]), np.array([False, False, False, False, False, False, False,  True]), np.array([False, False, False, False, False, False, False,  True]), np.array([False, False, False, False, False, False, False,  True])]})            
                  self.M -= 1
            elif (i == 4) and (self.deceptions[i] == 6) and (self.M > 0):
                    # print("deception on mssg happened i = 4")
                    msg = np.array([[False, False,  True], [False, False,  True], [False, False,  True], [False, False,  True]])
                    observation[agent][-32:] = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1], dtype='int')
                    dict_obs_all[agent].update({"message": [np.array([False, False, False, False, False, False, False,  True]), np.array([False, False, False, False, False, False, False,  True]), np.array([False, False, False, False, False, False, False,  True]), np.array([ False, False, False, False, False, False, False,  True])]})
                    self.M -= 1
            # print(f"after agant {i} was contaminated: {msg}")
            # Update the graph based on the raw dictionary 
            # print(f"agent {i} deceptions[i] is {self.deceptions[i]}")
            self.manp_obs = deepcopy(observation)
            o = observation[agent]
            g = self.graphs[agent]
            g.parse_observation(dict_obs)

            # Pull node features from tabular observation, and also update 
            # graph subnet connectivity edges. 
            tab_x,phase,new_msg = self._parse_tabular(o, g) 
            # tab_x,phase,new_msg = self._parse_tabular(o, g, deceptions=self.deceptions) 
            
            self.msg[agent] = new_msg

            # Combine node features from graph source, and tabular source
            x,ei,masks = g.get_state(MY_SUBNETS[i])
            x = self._combine_data(x, tab_x)

            # Mask/pack into conviniently sized tensors for the GNN models 
            obs = self._to_obs(x,ei,masks,phase,msg,new_msg)

            # During training, we need to know if the agent is still mid-action
            # If so, we don't bother calculating an action next turn 
            is_blocked = dict_obs['success'] == TernaryEnum.IN_PROGRESS
            graph_obs[agent] = (obs, is_blocked)

        self.ts += 1
        self.last_obs = graph_obs
        self.manp_dict_obs = dict_obs_all
        try:
            self.action_adv_list[:,self.reset_count,self.step_count] = np.copy(self.deceptions)
            self.action_def_list[:,self.reset_count,self.step_count] = np.array(list(action.values()), str)
            # self.observation_list.append()
            # print(reward)
            self.reward_list[self.reset_count, self.step_count] = np.mean(np.array(list(reward.values()), int))
            # self.mssg_list = []
            # self.mssg_dis_list = []
            self.budget[self.reset_count, self.step_count] = self.M
            self.step_count += 1
            # print(self.budget)
        except:
            pass
            # print("reset_cnt: " + str(self.reset_count))
            # print("step_cnt: " + str(self.step_count))
        return graph_obs, reward, term, trunc, info

    def reset(self):
        # print(self.reset_count) #always printed 0. in avaluation of episodes, not the same wrapper is used.
        self.reset_count += 1
        self.step_count = 0
        if self.reset_count > 98:
            # np.save("NoAttck_action_adv_list.npy", self.action_adv_list, allow_pickle=True)
            # np.save("NoAttck_action_def_list.npy", self.action_def_list, allow_pickle=True)
            # np.save("NoAttck_reward_list.npy", self.reward_list, allow_pickle=True)
            # np.save("NoAttck_budget.npy", self.budget, allow_pickle=True)

            # np.save("RandomAttck_action_adv_list.npy", self.action_adv_list, allow_pickle=True)
            # np.save("RandomAttck_action_def_list.npy", self.action_def_list, allow_pickle=True)
            # np.save("RandomAttck_reward_list.npy", self.reward_list, allow_pickle=True)
            # np.save("RandomAttck_budget.npy", self.budget, allow_pickle=True)

            # np.save("Agent0Attck_action_adv_list.npy", self.action_adv_list, allow_pickle=True)
            # np.save("Agent0Attck_action_def_list.npy", self.action_def_list, allow_pickle=True)
            # np.save("Agent0Attck_reward_list.npy", self.reward_list, allow_pickle=True)
            # np.save("Agent0Attck_budget.npy", self.budget, allow_pickle=True)

            # np.save("Agent1Attck_action_adv_list.npy", self.action_adv_list, allow_pickle=True)
            # np.save("Agent1Attck_action_def_list.npy", self.action_def_list, allow_pickle=True)
            # np.save("Agent1Attck_reward_list.npy", self.reward_list, allow_pickle=True)
            # np.save("Agent1Attck_budget.npy", self.budget, allow_pickle=True)

            # np.save("Agent2Attck_action_adv_list.npy", self.action_adv_list, allow_pickle=True)
            # np.save("Agent2Attck_action_def_list.npy", self.action_def_list, allow_pickle=True)
            # np.save("Agent2Attck_reward_list.npy", self.reward_list, allow_pickle=True)
            # np.save("Agent2Attck_budget.npy", self.budget, allow_pickle=True)

            # np.save("Agent3Attck_action_adv_list.npy", self.action_adv_list, allow_pickle=True)
            # np.save("Agent3Attck_action_def_list.npy", self.action_def_list, allow_pickle=True)
            # np.save("Agent3Attck_reward_list.npy", self.reward_list, allow_pickle=True)
            # np.save("Agent3Attck_budget.npy", self.budget, allow_pickle=True)

            # np.save("Agent4Attck_action_adv_list.npy", self.action_adv_list, allow_pickle=True)
            # np.save("Agent4Attck_action_def_list.npy", self.action_def_list, allow_pickle=True)
            # np.save("Agent4Attck_reward_list.npy", self.reward_list, allow_pickle=True)
            # np.save("Agent4Attck_budget.npy", self.budget, allow_pickle=True)

        
            # np.save("TurboAttck_action_adv_list.npy", self.action_adv_list, allow_pickle=True)
            # np.save("TurboAttck_action_def_list.npy", self.action_def_list, allow_pickle=True)
            # np.save("TurboAttck_reward_list.npy", self.reward_list, allow_pickle=True)
            # np.save("TurboAttck_budget.npy", self.budget, allow_pickle=True)

            np.save("SingleTurboAttck_action_adv_list.npy", self.action_adv_list, allow_pickle=True)
            np.save("SingleTurboAttck_action_def_list.npy", self.action_def_list, allow_pickle=True)
            np.save("SingleTurboAttck_reward_list.npy", self.reward_list, allow_pickle=True)
            np.save("SingleTurboAttck_budget.npy", self.budget, allow_pickle=True)
            # numpy.save("NewerAttck_action_adv_list.npy", action_adv_list, allow_pickle=True)
            # numpy.save("NewerAttck_action_adv_list.npy", action_adv_list, allow_pickle=True)
            # numpy.save("NewerAttck_action_adv_list.npy", action_adv_list, allow_pickle=True)
            llll = 0
        # if (self.reset_count % 3) == 0:
        #     print(self.action_adv_list)
        #     print()
        #     print(self.action_def_list)
        #     print()
        #     print(self.reward_list)
        #     print()
        #     print(self.budget)
        #     print()
            
        

        if self.FIRST_TIME:
            from submission import Submission
            self.FIRST_TIME = False
            self.submission = Submission()
        # print(f"past M before reset was: {self.M}")
        self.M = 500
        # print("reset called")
        '''
        Rebuild internal graph representation with parameters of new environment
        '''
        self.ts = 0
        """ should I have this? """
        

        obs_tab, action_mask = super().reset()
        # print(f"obs_tab in rest: {obs_tab}")
        self.gen_obs = deepcopy(obs_tab)
        self.manp_obs = deepcopy(obs_tab)
        g = ObservationGraph()

        # I don't *think* this is cheating, because FixedActionWrapper gets
        # to manipulate the obs returned by env.reset() which is the same thing.
        # Graph updates after intialization will all be using partial knowledge
        # known only to the agents.
        obs_dict = self.env.environment_controller.init_state
        self.manp_obs_dict = obs_dict

        g.setup(obs_dict)

        # Set message to empty for all agents
        self.msg = {
            a:np.zeros(8)
            for a in self.agent_names
        }

        my_state = dict()
        self.graphs = dict()
        for i in range(5):
            agent = f'blue_agent_{i}'
            o = obs_tab[agent]

            # Message from agent 4 has 2 extra subnet infos
            if i != 4:
                dummy_msg = (np.zeros((6,3)), np.zeros(8))
            else:
                dummy_msg = (np.zeros((4,3)), np.zeros(8))

            # Duplicate shared observation of the initial graph across
            # all agents (but make sure not to pass by reference)
            g_ = deepcopy(g)
            self.graphs[agent] = g_

            # Get tabular features and update connectivity graph 
            # tab_x,phase,_ = self._parse_tabular(o,g_,deceptions=[0, 0, 0, 0, 0])
            tab_x,phase,_ = self._parse_tabular(o,g_)

            # Combine all node features together and package for agents
            x,ei,masks = g_.get_state(MY_SUBNETS[int(agent[-1])])
            x = self._combine_data(x, tab_x)
            obs = self._to_obs(x,ei,masks,phase, *dummy_msg)
           
            # By default, agents are not blocked on turn 0

            # if action_mask:
            #     print(f"in reset my state is {my_state}")
            my_state[agent] = (obs, False)

        self.last_obs = my_state
        return my_state, action_mask


    def _parse_tabular(self, x, g):
    # def _parse_tabular(self, x, g, deceptions):
        '''
        Extract the per-host data from the tabular observation. 
        Data is formatted as 
        
        Mission phase           (1d, 0,1, or 2)
        Subnet info (x8)
            Subnet id           (9d)
            Blocked subnets     (9d)
            Comm policy         (9d)
            Comprimised hosts   (16d)
            Scanned hosts       (16d)
        Messages (4x8)

        We pull out the host data and append it to the host nodes, 
        pass the phase as a global vector, 
        and add messages to the subnet nodes they originate from 
        '''
        # First bit is phase
        # Last 4x8 are messages from other agents

        phase_idx = int(x[0])
        sn_block = x[1:-(4*8)]
        subnets = sn_block.shape[0] // SN_BLOCK_SIZE

        relevant_subnets = []
        src = []
        dst = []
        x = torch.zeros(g.n_permenant_nodes, 2)
        msgs = []

        # Only affects agent4 but may as well be generalizeable
        for i in range(subnets):
            block = sn_block[SN_BLOCK_SIZE*i : SN_BLOCK_SIZE*(i+1)]

            # Pull out edges between subnets
            sn = block[:18]
            me = ROUTERS[ sn[:9].nonzero()[0][0] ]
            can_maybe_connect_to = (sn[9:18] == 0).nonzero()[0]

            # Logic for subnet routing 
            if INTERNET in can_maybe_connect_to:
                can_connect_to = [ROUTERS[i] for i in can_maybe_connect_to]
            else:
                # Can connect to anything in LAN
                can_connect_to = [
                    ROUTERS[i] for i in can_maybe_connect_to
                    if ROUTERS[i] in ACCESSABLE_OFFLINE[me]
                ]

            router_name = me
            me = [me] * len(can_connect_to)
            src += can_connect_to
            dst += me

            # Pull out features for servers/hosts that exist
            hosts = torch.from_numpy(block[27:]).reshape(2,16).T
            n_srv, n_usr = g.subnet_size[router_name]
            srv_idx = list(range(n_srv))
            usr_idx = list(range(6,n_usr+6))

            # Insert into rows corresponding w server/host nodes in graph
            # (Always directly after node for subnet they are on)
            start_usr_idx = g.nids[router_name]+1
            start_srv_idx = start_usr_idx + len(usr_idx)
            end_srv_idx = start_srv_idx + len(srv_idx)

            # Note: TabularWrapper goes from server to host, but
            # graph goes from host to server (alphabetically)
            # so we have to do some lifting to rearrange
            x[start_usr_idx : start_srv_idx] = hosts[usr_idx]
            x[start_srv_idx : end_srv_idx] = hosts[srv_idx]

            # Each subnet can add 2 bits to the message for if any hosts
            # are compromised/have been scanned
            """ consider M + this "i" is not the agent number is subnet number. this manipulation shouldn't work """
            # if (deceptions[i] == 6) and self.M > 0:
            #     print("_parse_tabular deceives")
            #     msg = list((hosts.sum(dim=0) > 1).long())
            #     self.M -= 1                
            # else:
                # print(f"_parse_tabular not deceives with action {deceptions[i]}")
                # msg = list((hosts.sum(dim=0) > 0).long())
            msg = list((hosts.sum(dim=0) > 0).long())
            # print(f"raw message: {msg}")
            msgs += msg
            relevant_subnets.append(router_name)            
        # print(msgs)
        g.set_firewall_rules(src,dst)
        phase = torch.zeros((1,3))
        phase[0,phase_idx] = 1

        # Make messages all 8-dim and add checkbit to the end
        padding = 8-len(msgs)
        msgs += [0]*padding
        msgs[-1] = 1
        msg = np.array(msgs)
        # print(f"8-bit message: {msg}")
        return x, phase, msg

    def _combine_data(self, graph_x, tabular_x):
        '''
        Stick the tabular data onto the node feature matrix 
        on the appropriate rows--those corresponding with 
        the hosts the tabular data is referencing 
        '''
        # Tabular x only accounts for subnets and workstations
        # Processes/connections have higher indices, but no features
        # from the FlatActionWrapper, so need to be padded before combined
        padding = torch.zeros(
            graph_x.size(0) - tabular_x.size(0),
            tabular_x.size(1)
        )
        tabular_x = torch.cat([tabular_x, padding], dim=0)

        return torch.cat([graph_x, tabular_x], dim=1)

    def _to_obs(self, x,ei,masks,phase, other_msg,my_msg):
        '''
        Prepare for GNN injestion (assumes unbatched. E.g. this is
        called during inference, not training)

        Args:  
            x: feature matrix           (Nxd tensor)
            ei: edge index              (2xE tensor)
            masks: list of bitmaps for servers, users, 
                   subnet edges, and routers 
            phase: global state vector  (1x3 tensor)
            other_msg: messages from other agents
                                        (4x8 tensor)
            my_msg: the message this agent sends to 
                    other agents        (1x8 tensor)
        '''

        # Happens in all cases except agent_4
        if len(masks) == 1:
            (srv,usr,edge,rtrs) = masks[0]

            all_msg = torch.zeros((x.size(0), 3))
            all_msg[rtrs] = torch.from_numpy(other_msg).float()

            # Edge[0][0] is always the subnet node managed by this agent
            all_msg[edge[0][0], :2] = torch.from_numpy(my_msg[:2]).float()

            # Set 'is_recieved' to a special value to indicate this is self
            all_msg[edge[0][0], 2] = -1

            x = torch.cat([x, all_msg], dim=1)

            return (
                x,ei,phase,
                srv,torch.tensor([srv.size(0)]),
                usr,torch.tensor([usr.size(0)]),
                edge, False
            )

        # If this is agent 4's state, we have to be careful to batch 
        # three observations together. The graph info is the same, 
        # so we don't duplicate that, but we have to concat the 
        # masks together and scatter the messages we recieved
        # properly to the feature matrix
        srv,usr,edges = [],[],[]
        n_srv,n_usr = [],[]
        my_ids = []
        for (s,u,e,_) in masks:
            my_ids.append(e[0][0].item())

            srv.append(s)
            usr.append(u)
            edges.append(e)

            n_srv.append(s.size(0))
            n_usr.append(u.size(0))

        rtrs = masks[0][3]
        other_rtrs = [o.item() for o in rtrs if o.item() not in my_ids]
        all_msg = torch.zeros(x.size(0), 3)

        all_msg[other_rtrs] = torch.from_numpy(other_msg).float()
        all_msg[my_ids, :2] = torch.from_numpy(my_msg[:6].reshape(3,2)).float()
        all_msg[my_ids, 2] = -1

        x = torch.cat([x, all_msg], dim=1)

        return (
            x,ei,phase.repeat_interleave(3,0),
            torch.cat(srv), torch.tensor(n_srv),
            torch.cat(usr), torch.tensor(n_usr),
            torch.cat(edges, dim=1), True
        )

    def encode_dict_obs_expnd_inpt(self, dict_obs, gen_obs):
        # Map based on enum name strings
        success_map = {
            "TRUE":        [1, 0, 0, 0],
            "FALSE":       [0, 1, 0, 0],
            "UNKNOWN":     [0, 0, 1, 0],
            "IN_PROGRESS": [0, 0, 0, 1],
        }
        action = dict_obs.get('action', "Monitor")
        action_hashed = string_to_range(action)
        # print(f"action hasehd {action_hashed}")

        # Use enum.name for lookup
        success_enum = dict_obs.get('success', TernaryEnum.UNKNOWN)
        success_encoding = success_map.get(success_enum.name, [0, 0, 1, 0])  # fallback = UNKNOWN
    
        # Flatten the 4 message arrays
        message_vec = []
        for arr in dict_obs.get('message', []):
            message_vec.extend(arr.astype(np.float32).tolist())
    
        if len(message_vec) < 32:
            message_vec += [0.0] * (32 - len(message_vec))
    
        # Combine all features
        # features = success_encoding + message_vec
     

        # print(gen_obs)
        
        features = [action_hashed] + success_encoding + message_vec  + [float(self.M)] + gen_obs.tolist() 
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        # print("Feature tensor shape:", features_tensor.shape)
        
        return features_tensor
    


def string_to_range(s):
    """
    Maps a string deterministically to an integer in [0, N-1]
    """
    # print(s)
    h = hashlib.sha256(str(s).encode('utf-8')).hexdigest()
    # print("hash" + str(h))
    # print(int(h, 16))
    return (int(h, 16) % 250)/100

def encode_dict_obs(dict_obs):
    # Map based on enum name strings
    success_map = {
        "TRUE":        [1, 0, 0, 0],
        "FALSE":       [0, 1, 0, 0],
        "UNKNOWN":     [0, 0, 1, 0],
        "IN_PROGRESS": [0, 0, 0, 1],
    }

    # Use enum.name for lookup
    success_enum = dict_obs.get('success', TernaryEnum.UNKNOWN)
    success_encoding = success_map.get(success_enum.name, [0, 0, 1, 0])  # fallback = UNKNOWN

    # Flatten the 4 message arrays
    message_vec = []
    for arr in dict_obs.get('message', []):
        message_vec.extend(arr.astype(np.float32).tolist())

    if len(message_vec) < 32:
        message_vec += [0.0] * (32 - len(message_vec))

    # Combine all features
    features = success_encoding + message_vec
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)




