import os
from models.cage4 import load
# from models.cage4_ST import load

from CybORG import CybORG
from CybORG.Agents import BaseAgent

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from CybORG.Agents.Wrappers import EnterpriseMAE
# from wrapper.graph_wrapper import GraphWrapper
# from wrapper.graph_wrapper_main import GraphWrapper
# from wrapper.ST_wrapper import STGraphWrapper

# from wrapper.modified_graph_wrapper import MGraphWrapper
# from wrapper.modified_multi_graph_wrapper import MMGraphWrapper
# from wrapper.modified_ray_graph_wrapper import MRGraphWrapper
# from wrapper.modified_multi_const_graph_wrapper import MCGraphWrapper
# from wrapper.newattack_modified_multi_const_graph_wrapper import NMCGraphWrapper
# from wrapper.newerattack_modified_multi_const_graph_wrapper import NMCGraphWrapper
from wrapper.turbo_wrapper import TGraphWrapper
# from wrapper.turbo_wrapper_greedy import TGraphWrapper
# from wrapper.single_turbo_wrapper import STGraphWrapper


# from wrapper.random_modified_multi_const_graph_wrapper import RNMCGraphWrapper


### Import custom agents here ###
# from dummy_agent import DummyAgent
# from TrainedBlueAgent import TrainedBlueAgent


class Submission:

    # Submission name
    # NAME: str = "TEST TrainedBlueAgent"
    NAME: str = "KEEP"

    # Name of your team
    # TEAM: str = "CyberArmit"
    # TEAM: str = "Cybermonic"
    TEAM: str = "Cybermonic-Modified"



    # What is the name of the technique used? (e.g. Masked PPO)
    # TECHNIQUE: str = "PPO GPT"
    TECHNIQUE: str = "Graph-based PPO With Intra-agent Communication"

    # Use this function to define your agents.
    # AGENTS = {
        # f"blue_agent_{i}": load(f'{os.path.dirname(__file__)}/weights/policy_state_{i}.pkl')
        # for i in range(5)
    # }
    # AGENTS: dict[str, BaseAgent] = {
    #     f"blue_agent_{agent}": TrainedBlueAgent(f"Agent{agent}", np_random=10) for agent in range(5)
    # }
    AGENTS = {
        f"blue_agent_{i}": load(f'{os.path.dirname(__file__)}/weights/gnn_ppo-{i}.pt')
        for i in range(5)
    }
    # Use this function to wrap CybORG with your custom wrapper(s) optionally.
    # def wrap(env: CybORG) -> MultiAgentEnv:
    #     return EnterpriseMAE(env=env)


    # Use this function to optionally wrap CybORG with your custom wrapper(s).
    def wrap(env: CybORG) -> MultiAgentEnv:
        # return GraphWrapper(env)
        # return STGraphWrapper(env)
        # return MGraphWrapper(env)
        # return MMGraphWrapper(env)
        # return MCGraphWrapper(env)
        # return NMCGraphWrapper(env)
        return TGraphWrapper(env)
        # return STGraphWrapper(env)
        # return RNMCGraphWrapper(env)