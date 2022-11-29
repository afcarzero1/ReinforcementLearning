from MazeMDP.Modeling.mdp_modeling import MDP
import numpy as np


def qLearningSolver(mdp : MDP):

    # Initialize the matrix Q
    number_states : int = len(mdp.states())
    numer_actions : int = len(mdp.all_actions())

    q_matrix = np.zeros((number_states, numer_actions))

    pass