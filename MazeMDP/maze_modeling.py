import numpy as np
from termcolor import colored

from mapbuilders import build_simple_map, build_blueberries_map
from mdp_modeling import MDPTerminalState, dynamicProgramSolver, printPath, print_successors


class MapProblemNotChooseWall(MDPTerminalState):
    """
    This class represents a map problem in which the player cannot move to a cell in which there is a wall
    """

    def __init__(self, map_layout: np.ndarray, map_rewards: np.ndarray, final_state: (int, int),
                 start_state: (int, int) = (0, 0),stay_enabled=True):
        self._action_map_built = False
        self._final_state = final_state
        self._initial_state = start_state
        self.stay_enabled = stay_enabled

        # Define the problem statically
        self.map_rewards = map_rewards
        self._define_problem(map_layout)

    def _define_problem(self, map_layout: np.ndarray):
        """
        This function defines the problem from a map layout.
        Args:
            map_layout ( np.ndarray ): This is  the layout of the map. A value of 1 symbolizes a wall.
        :return:
        """

        self._map = map_layout

        # Create all the possible states
        self._states = [(x, y) for x in range(0, map_layout.shape[0]) for y in range(0, map_layout.shape[1])]
        self._actions = ['up', 'dw', 'l', 'r', 's']

        # Extra information useful for not repeating process
        self.state_to_action = {}

        # Compute for every state the available actions for the player
        for state in self._states:
            state_actions = self.actions(state)
            self.state_to_action[state] = state_actions

        # Set the action map built bit to True
        self._action_map_built = True

    def actions(self, state):
        # Return the actions if already built
        if self._action_map_built:
            return self.state_to_action[state]

        actions = self._actions.copy()

        ## Check map boundaries and walls
        ## UP
        if state[0] <= 0:
            actions.remove('up')
        elif self._map[state[0] - 1, state[1]] == 1:
            actions.remove('up')

        # DOWN
        if state[0] >= (self._map.shape[0] - 1):
            actions.remove('dw')
        elif self._map[state[0] + 1, state[1]] == 1:
            actions.remove('dw')

        # LEFT
        if state[1] <= 0:
            actions.remove('l')
        elif self._map[state[0], state[1] - 1] == 1:
            actions.remove('l')

        # RIGHT
        if state[1] >= (self._map.shape[1] - 1):
            actions.remove('r')
        elif self._map[state[0], state[1] + 1] == 1:
            actions.remove('r')

        # TERMINAL : Only stay in cell actio  is available
        if self.is_end(state):
            actions = ['s'] if self.stay_enabled else []

        return actions

    def states(self):
        return self._states

    def discount(self):
        return 1

    def successor_prob_reward(self, state, action):
        next_state = (0, 0)
        prob = 1

        if action == 'up':
            next_state = (state[0] - 1, state[1])
        elif action == 'dw':
            next_state = (state[0] + 1, state[1])
        elif action == 'l':
            next_state = (state[0], state[1] - 1)
        elif action == 'r':
            next_state = (state[0], state[1] + 1)
        elif action == 's':
            next_state = state

        reward = self.map_rewards[next_state]
        return [(next_state, prob, reward)]

    def start_state(self):
        return self._initial_state

    def is_end(self, state):
        return state == self._final_state

    def print_values(self, values: dict):

        for i in range(self._map.shape[0]):
            for j in range(self._map.shape[1]):
                tup = (i, j)
                print("{:5.2f}".format(values[tup]), end=" ")
            print()

    def print_actions(self, actions: dict):
        for i in range(self._map.shape[0]):
            for j in range(self._map.shape[1]):
                tup = (i, j)
                str_to_print = actions[tup] if actions[tup] is not None else "None"
                print("{:>5}".format(str_to_print), end=" ")
            print()


class MapProblemRandom(MapProblemNotChooseWall):
    def successor_prob_reward(self, state, action):
        succ, prob, rew = super(MapProblemRandom, self).successor_prob_reward(state, action)[0]

        if succ == (5, 0):
            to_return = []
            bad_reward = self.map_rewards[succ] * 7
            bad_reward_prob = 1
            normal_reward = self.map_rewards[succ]
            normal_reward_prob = 1 - bad_reward_prob
            to_return.append((succ, bad_reward_prob, bad_reward))
            to_return.append((succ, normal_reward_prob, normal_reward))

            return to_return

        if succ == (4, 6):
            to_return = []
            bad_reward = self.map_rewards[succ] * (1 + 1)
            bad_reward_prob = 0.5
            normal_reward = self.map_rewards[succ]
            normal_reward_prob = 1 - bad_reward_prob
            to_return.append((succ, bad_reward_prob, bad_reward))
            to_return.append((succ, normal_reward_prob, normal_reward))

            return to_return

        return [(succ,prob,rew)]


class MapProblemMinotaur(MapProblemNotChooseWall):
    pass




def solve_map(problem: MapProblemNotChooseWall, time_horizon=25):
    print(colored("THE MAP IS :", 'blue'))
    print(problem._map)
    print(colored("The states are", 'blue'))
    print(problem.states())
    _, level_actions = dynamicProgramSolver(problem, time_horizon)
    print(colored("The path from initial state is", 'blue'))
    print(printPath(problem, level_actions))


def main():
    # Build the MDP with a simple map
    #map_layout, map_rewards, final_state = build_simple_map()
    map_layout, map_rewards, final_state = build_blueberries_map()

    mdp = MapProblemNotChooseWall(map_layout, map_rewards, final_state,stay_enabled=False)
    #mdp = MapProblemRandom(map_layout,map_rewards,final_state,stay_enabled=False)

    ## Run a test to see if it is working

    for state in mdp.states():
        print_successors(mdp, state)

    time_horizon = 25
    solve_map(mdp, time_horizon)


if __name__ == '__main__':
    main()
