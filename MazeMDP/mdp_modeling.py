"""
Module modeling MDP problems
"""

import numpy as np
from abc import ABC, abstractmethod
from termcolor import colored


class MDP(ABC):
    @abstractmethod
    def actions(self, state):
        pass

    @abstractmethod
    def states(self):
        pass

    @abstractmethod
    def start_state(self):
        return 0, 0

    @abstractmethod
    def successor_prob_reward(self, state, action):
        pass

    def discount(self):
        return 1

    @abstractmethod
    def print_values(self, values: dict):
        pass

    @abstractmethod
    def print_actions(self, actions: dict):
        pass


class MDPTerminalState(MDP):
    @abstractmethod
    def is_end(self, state):
        pass


def dynamicProgramSolver(problem: MDP, simulation_time: int = 5):
    T = simulation_time

    remaining_reward = {}  # Dictionary (state -> average remaining reward at a given level)
    best_action = {}

    level_rewards = {}
    level_actions = {}

    def Q(state, action):
        su = sum(prob * (reward + problem.discount() * remaining_reward[newState]) \
                 for newState, prob, reward in problem.successor_prob_reward(state, action))
        return su

    for level in range(T, 0, -1):
        print(colored(f"Level {level}", 'blue'))

        if level == T:
            # maximize over reachable states (greedy solution)
            for state in problem.states():
                maximum_reward_state = 0
                best_state_action = None
                # Compute the maximum reward reachable from this state. Evaluate all the possible actions we can take
                # in this state. Take the action with the maximum possible reward (there might be several)
                for action in problem.actions(state):
                    # For a given action we can end up in several states with a certain probability.
                    # We choose the action that leads to the best
                    rewards = [reward for next_state, prob, reward in problem.successor_prob_reward(state, action)]
                    max_reward_action = max(rewards)
                    if max_reward_action > maximum_reward_state:
                        maximum_reward_state = max_reward_action
                        best_state_action = action

                # Update the remianing reward

                remaining_reward[state] = maximum_reward_state
                best_action[state] = best_state_action

            level_rewards[level] = remaining_reward.copy()
            level_actions[level] = best_action.copy()

            # print(remaining_reward)
        else:
            # Apply the recursion algorithm
            next_remaining_reward = {}  # (state -> remaining reward at level)
            for state in problem.states():
                maximum_reward_state = 0
                best_state_action = None
                for action in problem.actions(state):
                    # Examine for the action the possible states it is possible to end up in. Iterate over them using
                    # the probability to reach them and the remaining reward computed for the level+1
                    max_reward_action = Q(state, action)

                    # Substitute the best action if we obtain a higher average remaining reward
                    if max_reward_action > maximum_reward_state:
                        maximum_reward_state = max_reward_action
                        best_state_action = action

                # Save it
                next_remaining_reward[state] = maximum_reward_state
                best_action[state] = best_state_action
            # Update the remaining values
            remaining_reward = next_remaining_reward.copy()

            level_rewards[level] = remaining_reward.copy()
            level_actions[level] = best_action.copy()
            # print(remaining_reward)

        # Print for having an idea
        problem.print_values(remaining_reward)
        problem.print_actions(best_action)

    return level_rewards, level_actions


def printPath(problem: MDP, solution: dict):
    min_level = min(solution.keys())
    max_level = max(solution.keys())

    decisions = []
    state = problem.start_state()
    collected_reward = 0
    for level in range(min_level, max_level+1):
        level_decisions = solution[level][state]
        decisions.append((state, level_decisions))

        if level_decisions is None:
            break

        state,_,level_reward = problem.successor_prob_reward(state, level_decisions)[0] #todo : understand how to deal with this
        collected_reward +=level_reward

    print(collected_reward)
    return decisions


def print_successors(mdp : MDP,state : (int,int)):
    actions = mdp.actions(state)
    for action in actions:
        for succ,prob,rew in mdp.successor_prob_reward(state,action):
            print(f"[STATE {state}] [Action {action}] [Successor {succ}] : {prob} , {rew}")


def main():
    pass


if __name__ == '__main__':
    main()
