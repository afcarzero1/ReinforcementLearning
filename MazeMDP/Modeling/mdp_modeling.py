"""
Module modeling MDP problems in a general way
"""
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import numpy as np
from gym import Env


class MDP(ABC, Env):
    internal_state = None

    @abstractmethod
    def actions(self, state):
        r"""
        Return the actions available for a state
        Args:
            state (Any) : State for which actions must be returned
        Return:
            actions ([Any]) : List with the actions for the given state
        """
        pass

    @abstractmethod
    def states(self) -> List[Any]:
        r"""
        Return all the states of the Markov Decision Process
        Return:
            states ([Any]) : List with the states of the Markov Decision Process
        """
        pass

    @abstractmethod
    def start_state(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def successor_prob_reward(self, state: Any, action: Any) -> List[Tuple[Any, float, float]]:
        r"""
        Return in a list all the possible successors of the given state taking action with their respective probabilities and rewards.
        Returned tuples have the format (state',Probability(state,action,state'),Reward(state,state'))

        Args:
            state(Any) : Starting state
            action (Any) : Action taken
        Return:
            successors_probs_rewards ([(Any,float,float)] : List with tuples representing the successor , its probability and the reward
        """
        pass

    def all_actions(self):
        raise NotImplementedError

    def discount(self) -> float:
        r"""
        Return the discount factor of the Markov Decision Process

        Return:
            Floating point number between 0 and 1 representing the discount factor
        """
        return 1.0

    @abstractmethod
    def print_values(self, values: dict):
        pass

    @abstractmethod
    def print_actions(self, actions: dict):
        pass

    def is_goal(self, state : Any) -> bool:
        r"""
        Function indicating if the MDP state is a "positive" result.
        Args:
            state(Any) : State of the MDP to anlayze.
        """
        pass

    def reset(self, **kwargs) -> Any:
        r"""
        Reset the internal state of the Markov Decision Process to the initial state.
        Return:
             state(Any) : The state in which the MDP is now
        """
        super(MDP, self).reset(**kwargs)
        self.internal_state = self.start_state()
        return self.internal_state

    def step(self, action) -> Tuple[Any, float, bool, bool, dict]:
        r"""
        Take a step using the action
        Args:
            action(Any) : action to take
        Return:
            next_state_prob_reward
        """
        list_possible_states = self.successor_prob_reward(self.internal_state, action)
        probs: [float] = [prob for next_state, prob, reward in list_possible_states]

        choice: int = np.random.choice(np.arange(0, len(list_possible_states)), p=probs)
        self.internal_state = list_possible_states[choice][0]

        succ, prob, rew = list_possible_states[choice]
        return succ, rew, False, False, {"probabaility": prob}

    def render(self):
        pass


class MDPTerminalState(MDP):
    r"""
    Class representing a Markov Decision Process with a terminal state
    """

    def step(self, action) -> (Any, float, bool, bool, dict):
        succ, rew, ter, trun, d = super(MDPTerminalState, self).step(action)
        terminal = self.is_end(succ)
        return succ, rew, terminal, trun, d

    @abstractmethod
    def is_end(self, state: Any) -> bool:
        r"""
        Determine if the given state is terminal
        Args:
            state(Any) : State to determine if it is terminal
        """
        raise NotImplementedError


def print_successors(mdp: MDP, state: Any):
    """
    Print the successors of a given state for the given MDP.
    Args:
        mdp (MDP) : Markov Decision Process
        state (Any) : The state to analyze
    """
    actions = mdp.actions(state)
    for action in actions:
        for succ, prob, rew in mdp.successor_prob_reward(state, action):
            print(f"[STATE {state}] [Action {action}] [Successor {succ}] : {prob} , {rew}")


def save_average_reward(mdp : MDP, path : str = "./average_rewards.txt"):
    with open(path,"w") as f:
        for state in mdp.states():
            tot_reward = 0
            for action in mdp.actions(state):
                for succ, prob, rew in mdp.successor_prob_reward(state, action):
                    tot_reward += prob * rew
                if tot_reward != 0:
                    f.write(str(state) + " " +str(action)+" "+ str(tot_reward) + '\n')
            tot_reward /= len(mdp.actions(state))


def main():
    pass


if __name__ == '__main__':
    main()
