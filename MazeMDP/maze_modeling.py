import numpy as np
from termcolor import colored

from mapbuilders import build_simple_map, build_blueberries_map, build_minotaour_map
from mdp_modeling import MDPTerminalState, dynamicProgramSolver, printPath, print_successors


class MapProblemNotChooseWall(MDPTerminalState):
    """
    This class represents a map problem in which the player cannot move to a cell in which there is a wall
    """

    def __init__(self, map_layout: np.ndarray, map_rewards: np.ndarray, final_state: (int, int),
                 start_state: (int, int) = (0, 0), stay_enabled=True):
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

        return [(succ, prob, rew)]


class MapProblemMinotaur(MapProblemNotChooseWall):
    def __init__(self, minotaur_position: (int, int), *args, **kwargs):
        super(MapProblemMinotaur, self).__init__(*args, **kwargs)
        self.minotaur_position = minotaur_position

    def _define_problem(self, map_layout: np.ndarray):
        self._map = map_layout

        # Define the states as a tuple ( x_player, y_player, x_mino, y_mino)
        self._states = [(x, y, x2, y2) for x in range(0, map_layout.shape[0]) for y in range(0, map_layout.shape[1])
                        for x2 in range(0, map_layout.shape[0]) for y2 in range(0, map_layout.shape[1])]
        self._actions = ['up', 'dw', 'l', 'r', 's']

        self.state_to_action = {}

        for state in self._states:
            state_actions = self.actions((state[0], state[1]))
            # If the minotaur cath you there is no action available (lot of future loss)
            if state[0] == state[2] and state[1] == state[3]:
                self.state_to_action[state] = ['s']
            else:
                self.state_to_action[state] = state_actions

        self._action_map_built = True

    def is_goal(self, state):
        return (state[0], state[1]) == self._final_state and not (state[0] == state[2] and state[1] == state[3])

    def successor_prob_reward(self, state, action):
        next_state_player = (0, 0)

        # Check wether they are in the same cell. This is final and bad
        if state[0] == state[2] and state[1] == state[3]:
            return [(state, 1, -200)]

        # Check if the player arrived and not move minotaur anymore (not possible to loose). Notice we already checked before
        # that they are not in the same cell
        if (state[0], state[1]) == self._final_state:
            return [(state, 1, 100)]

        # Update player state
        if action == 'up':
            next_state_player = (state[0] - 1, state[1])
        elif action == 'dw':
            next_state_player = (state[0] + 1, state[1])
        elif action == 'l':
            next_state_player = (state[0], state[1] - 1)
        elif action == 'r':
            next_state_player = (state[0], state[1] + 1)
        elif action == 's':
            next_state_player = (state[0], state[1])

        # Update minotaur state
        mino_states = self.minotaur_succ_prob((state[2], state[3]))

        # Combine player and mino state
        total_state = [(next_state_player + next_mino_state, prob,
                        -200 if next_state_player == next_mino_state else self.map_rewards[next_state_player])
                       for next_mino_state, prob in mino_states]

        # Compute the reward
        return total_state

    def minotaur_actions(self, state):

        minotaur_actions_list = ['up', 'dw', 'l', 'r','s']

        if state[0] <= 0:
            minotaur_actions_list.remove('up')
        if state[0] >= (self._map.shape[0] - 1):
            minotaur_actions_list.remove('dw')
        if state[1] <= 0:
            minotaur_actions_list.remove('l')
        if state[1] >= (self._map.shape[1] - 1):
            minotaur_actions_list.remove('r')

        return minotaur_actions_list

    def minotaur_succ_prob(self, state):
        mino_actions = self.minotaur_actions(state)

        mino_succ = []
        prob = 1 / len(mino_actions)

        next_state = (0, 0)
        for action in mino_actions:
            if action == 'up':
                next_state = (state[0] - 1, state[1])
            elif action == 'dw':
                next_state = (state[0] + 1, state[1])
            elif action == 'l':
                next_state = (state[0], state[1] - 1)
            elif action == 'r':
                next_state = (state[0], state[1] + 1)
            elif action == 's':
                next_state = (state[0],state[1])

            mino_succ.append((next_state, prob))
        return mino_succ

    def start_state(self):
        return self._initial_state + self.minotaur_position

    def print_actions(self, actions: dict):
        for i in range(self._map.shape[0]):
            for j in range(self._map.shape[1]):
                tup = (i, j) + self.minotaur_position
                str_to_print = actions[tup] if actions[tup] is not None else "None"
                print("{:>5}".format(str_to_print), end=" ")
            print()

    def print_values(self, values: dict):
        for i in range(self._map.shape[0]):
            for j in range(self._map.shape[1]):
                tup = (i, j) + self.minotaur_position
                print("{:5.2f}".format(values[tup]), end=" ")
            print()

    def render(self):
        # Print player position and minotaur
        player_pos = (self.internal_state[0], self.internal_state[1])
        minotaur_pos = (self.internal_state[2], self.internal_state[3])
        for i in range(self._map.shape[0]):
            for j in range(self._map.shape[1]):
                if (i,j) == player_pos and (i,j) == minotaur_pos:
                    print(colored("PD", "red"), end=" ")
                elif (i, j) == player_pos:
                    print(colored("P", "blue"), end=" ")
                elif (i, j) == minotaur_pos:
                    print(colored("M", 'red'), end=" ")
                else:
                    print("o", end=" ")

            print()


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
    # map_layout, map_rewards, final_state = build_simple_map()
    map_layout, map_rewards, final_state = build_blueberries_map()

    mdp = MapProblemNotChooseWall(map_layout, map_rewards, final_state, stay_enabled=False)
    # mdp = MapProblemRandom(map_layout,map_rewards,final_state,stay_enabled=False)

    ## Run a test to see if it is working

    for state in mdp.states():
        print_successors(mdp, state)

    time_horizon = 25
    solve_map(mdp, time_horizon)


def simulate_game(mdp: MDPTerminalState, level_actions: dict, verbose=False):
    min_level = min(level_actions.keys())
    max_level = max(level_actions.keys())
    current_state = mdp.reset()
    if verbose:
        mdp.render()

    collected_reward = 0
    for time in range(min_level, max_level + 1):

        # take decision according to MDP
        decision = level_actions[time][current_state]

        # Use action and see next state
        next_state, prob, reward = mdp.step(decision)
        # Collect the reward
        collected_reward += reward

        if verbose:
            print(f"{current_state} -> {next_state} : {reward}")
            mdp.render()

        current_state = next_state

    success = True if mdp.is_goal(current_state) else False
    return success,collected_reward


def minotaur():
    map_layout, map_rewards, final_state = build_minotaour_map()

    # Check that the map is solvable without the minotaur
    """
    mdp = MapProblemNotChooseWall(map_layout, map_rewards, final_state, stay_enabled=True)
    for state in mdp.states():
        print_successors(mdp, state)

    time_horizon = 25
    solve_map(mdp, time_horizon)
    """

    mdp = MapProblemMinotaur((6, 5), map_layout, map_rewards, final_state, stay_enabled=True)
    print(len(mdp.states()))
    # for state in mdp.states():
    # print_successors(mdp, state)
    print(map_layout)
    level_rewards, level_actions = dynamicProgramSolver(mdp, 16)


    # Show couple of games
    simulate_game(mdp, level_actions, verbose=True)


    successes = 0
    episodes = 1000
    total_reward = 0
    for _ in range(0,episodes):
        success,reward=simulate_game(mdp, level_actions, verbose=False)
        if success:
            successes+=1
        total_reward+=reward

    successes = successes/episodes
    total_reward = total_reward / episodes

    print(f"SUCCESS RATE : {successes}")
    print(f"Avg Reward : {total_reward}")


if __name__ == '__main__':
    # main()
    minotaur()
