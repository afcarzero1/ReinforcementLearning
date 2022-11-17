import time
from typing import Any

from termcolor import colored

from MazeMDP.mapbuilders import build_minotaour_map
from MazeMDP.maze_modeling import simulate_game, simulate_infinte_game
from MazeMDP.mdp_modeling import MDPTerminalState, dynamicProgramSolver, print_successors, valueIterationSolver
import numpy as np

class MapProblemMinotaurKeys(MDPTerminalState):

    def __init__(self, map_layout: np.ndarray, start_player: (int, int) = (0, 0),
                 start_minotaur: (int, int) = (5, 5)
                 , start_keys: bool = False, end_cell: (int, int) = (5, 5),
                 key_cell: (int, int) = (0, 5), allow_minotaur_stay: bool = True,
                 probability_uniform : float = 0.5,discount_factor:float=1.0):



        self._map = map_layout
        self._actions = ['up', 'dw', 'l', 'r', 's']
        self._allow_minotaur_stay = allow_minotaur_stay
        self._probability_uniform = probability_uniform
        self._discount_factor = discount_factor

        # Build the start state
        self._start_state = start_player + start_minotaur + (1,) if start_keys else (0,)
        self._end_cell = end_cell
        self._key_cell = key_cell

        ## Reward constants
        self._reward_caught = -1
        self._reward_arrived = 1
        self._reward_keys = 1
        self._reward_arriving = 1

        ## Render constant
        self.minotaur_position = (5,5)

        self._define_problem()

    def _define_problem(self):

        # Build the possible states of the problem
        self._states = [(x, y, x2, y2, k) for x in range(0, self._map.shape[0]) for y in range(0, self._map.shape[1])
                        for x2 in range(0, self._map.shape[0]) for y2 in range(0, self._map.shape[1])
                        for k in range(0, 2)]

        # Constant with the available actions for the player
        self._actions = ['up', 'dw', 'l', 'r', 's']

        self.player_state_to_action = {}
        self.minotaur_state_to_action = {}

        # Build for each state the possible actions
        for state in self._states:
            state_actions = self._player_actions((state[0], state[1]), check_walls=True)
            minotaur_actions = self._player_actions((state[2], state[3]), check_walls=False,
                                                    allow_stay=self._allow_minotaur_stay)
            self.minotaur_state_to_action[(state[2], state[3])] = minotaur_actions
            # Restrict the action space (equivalent to changing transition matrix)
            if state[0] == state[2] and state[1] == state[3]:
                self.player_state_to_action[state] = ['s']
            else:
                self.player_state_to_action[state] = state_actions


    def _player_actions(self, player_state: (int, int), check_walls=True, allow_stay=True):
        actions = self._actions.copy()

        ## Check map boundaries and walls
        ## UP
        if player_state[0] <= 0:
            actions.remove('up')
        elif self._map[player_state[0] - 1, player_state[1]] == 1 and check_walls:
            actions.remove('up')

        # DOWN
        if player_state[0] >= (self._map.shape[0] - 1):
            actions.remove('dw')
        elif self._map[player_state[0] + 1, player_state[1]] == 1 and check_walls:
            actions.remove('dw')

        # LEFT
        if player_state[1] <= 0:
            actions.remove('l')
        elif self._map[player_state[0], player_state[1] - 1] == 1 and check_walls:
            actions.remove('l')

        # RIGHT
        if player_state[1] >= (self._map.shape[1] - 1):
            actions.remove('r')
        elif self._map[player_state[0], player_state[1] + 1] == 1 and check_walls:
            actions.remove('r')

        if not allow_stay:
            actions.remove('s')

        return actions

    def actions(self, state):
        return self.player_state_to_action[state]

    def states(self):
        return self._states

    def start_state(self):
        return self._start_state

    def successor_prob_reward(self, state: Any, action: Any) -> [(Any, float, float)]:

        # Check if the minotaur has catched the player
        if state[0] == state[2] and state[1] == state[3]:
            return [(state, 1, self.reward(state,action,state))]

        # Check if the player arrived to the goal
        if (state[0], state[1]) == self._end_cell and state[4] == 1:
            return [(state, 1, self.reward(state,action,state))]

        # Compute the next state of the player
        next_player_state: tuple = self.deterministic_f((state[0], state[1]), action)

        # Compute the next state of the minotaur
        next_minotaur_states: [tuple] = self.minotaur_succ_prob(state)

        # Compute if the agent gets the key or not
        next_key = 1 if next_player_state == self._key_cell or state[4] == 1 else 0

        # (next_state, prob, reward)
        total_state = [(next_player_state + next_mino_state + (next_key,),
                        prob,
                        self.reward(state, action, next_player_state + next_mino_state + (next_key,))
                        )
                       for next_mino_state, prob in next_minotaur_states]

        return total_state

    def reward(self, state, action, state_arrival):

        state_player = (state[0], state[1])
        state_mino = (state[2], state[3])
        holding_keys = state[4]



        # Initial state we already have been caught
        if state_player == state_mino:
            return self._reward_caught

        # Initial state we already arrived without being caught , having the keys
        if state_player == self._end_cell and holding_keys==1:
            return self._reward_arrived

        # Generally moving and the next move arrives to the key
        if (holding_keys == 0 and state_arrival[4] == 1):
            return self._reward_keys

        if (state_arrival[0],state_arrival[1]) == self._end_cell and holding_keys==1:
            return self._reward_arriving

        # Else return the reward of moving to the next cell
        return 0

    def minotaur_succ_prob(self, state: (int, int, int, int, int)):
        # Compute uniform moves
        mino_state = (state[2], state[3])

        # If we catched the player do not move
        if state[0] == state[2] and state[1] == state[3]:
            return [(mino_state,1)]
        mino_actions = self.minotaur_state_to_action[mino_state]
        prob = 1 / len(mino_actions) * self._probability_uniform

        mino_succ_uni = [(self.deterministic_f(mino_state, action), prob) for action in mino_actions]

        # Compute moves towards player

        mino_succ_towards = [(self.deterministic_f(mino_state, self.g_delta(state)), 1 - self._probability_uniform)]

        # Combine all the possibilities
        mino_succ = mino_succ_uni + mino_succ_towards

        return mino_succ

    def deterministic_f(self, state: (int, int), action):
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
        else:
            raise ValueError

        return next_state_player

    def g_delta(self, state):

        # Compute the delta

        delta_y = state[0] - state[2]
        delta_x = state[1] - state[3]

        sgn_delta_x = np.sign(delta_x)
        sgn_delta_y = np.sign(delta_y)

        # Compute the g function
        if delta_y == 0 and delta_x == 0:
            return 's' #if self._allow_minotaur_stay else 'r'

        if abs(delta_x) > abs(delta_y):
            # Move in the y direction
            if sgn_delta_y == 1:
                return 'dw'
            elif sgn_delta_y == -1:
                return 'up'
            else:
                return 'l' if sgn_delta_x == -1 else 'r'

        if abs(delta_x) <= abs(delta_y):
            # Move in the x direction by default
            if sgn_delta_x == 1:
                return 'r'
            elif sgn_delta_x == -1:
                return 'l'
            else:
                return 'up' if sgn_delta_y == -1 else 'dw'

    def discount(self) -> float:
        return self._discount_factor

    def print_actions(self, actions: dict):
        for i in range(self._map.shape[0]):
            for j in range(self._map.shape[1]):
                tup = (i, j) + self.minotaur_position + (0,)
                str_to_print = actions[tup] if actions[tup] is not None else "None"
                if self._map[(i, j)] == 1:
                    str_to_print = "X"

                print("{:>5}".format(str_to_print), end=" ")
            print(" ", end=colored("|", 'red'))
            for j in range(self._map.shape[1]):
                tup = (i, j) + self.minotaur_position + (1,)
                str_to_print = actions[tup] if actions[tup] is not None else "None"
                if self._map[(i, j)] == 1:
                    str_to_print = "X"

                print("{:>5}".format(str_to_print), end=" ")
            print()

    def print_values(self, values: dict):
        for i in range(self._map.shape[0]):
            for j in range(self._map.shape[1]):
                tup = (i, j) + self.minotaur_position + (0,)
                if self._map[(i, j)] == 1:
                    str_to_print = "X"
                    print("{:5}".format(str_to_print),end=" ")
                else:
                    print("{:5.2f}".format(values[tup]), end=" ")
            print(" ",end=colored("|",'red'))
            for j in range(self._map.shape[1]):
                tup = (i, j) + self.minotaur_position + (1,)
                if self._map[(i, j)] == 1:
                    str_to_print = "X"
                    print("{:5}".format(str_to_print),end=" ")
                else:
                    print("{:5.2f}".format(values[tup]), end=" ")


            print()

    def render(self):
        # Print player position and minotaur
        player_pos = (self.internal_state[0], self.internal_state[1])
        minotaur_pos = (self.internal_state[2], self.internal_state[3])
        for i in range(self._map.shape[0]):
            for j in range(self._map.shape[1]):
                if (i, j) == player_pos and (i, j) == minotaur_pos:
                    print(colored("PD", "red"), end=" ")
                elif (i, j) == player_pos:
                    print(colored("P", "blue"), end=" ")
                elif (i, j) == minotaur_pos:
                    print(colored("M", 'red'), end=" ")
                else:

                    if self._map[(i, j)] == 1:
                        print(colored("x", 'green'), end=" ")
                    else:
                        print("o", end=" ")

            print()

    def is_end(self, state: Any) -> bool:
        return (state[0],state[1]) == self._end_cell or (state[0]==state[2] and state[1]==state[3])

    def is_goal(self,state : Any) -> bool:
        return (state[0],state[1]) == self._end_cell and not (state[0]==state[2] and state[1]==state[3])


PRINT_LENGTH = 100

def main():

    t=time.time()
    # 1.c
    #solve_minotaur_finite_horizon_with_keys()

    print(time.time()-t)
    t = time.time()
    # 1.d
    solve_minotaur_infinite_horizon_with_keys()
    print(time.time()-t)




def solve_minotaur_finite_horizon_with_keys():
    # Generate the map layout
    map_layout, _, final_state = build_minotaour_map(cell_reward=0)

    # Create the MDP
    mdp = MapProblemMinotaurKeys(map_layout=map_layout,
                                 start_player=(0, 0),
                                 start_minotaur=(6, 6),
                                 start_keys=True,
                                 end_cell=final_state,
                                 key_cell=(0, 7),
                                 allow_minotaur_stay=False,
                                 probability_uniform=1)

    # See number of states and map layout
    print(len(mdp.states()))
    print(map_layout)

    # Print the successors
    print(colored("-" * 25 + "STATES" + "-" * 25, 'red'))

    for _ in range(10):
        state_example = mdp.states()[np.random.randint(0, len(mdp.states()))]
        print(colored("-" * 50, 'red'))
        print_successors(mdp, state_example)

    print(colored("-" * 50, 'red'))
    print_successors(mdp, (6, 5, 5, 5, 0))
    print(colored("-" * 50, 'red'))
    print_successors(mdp, (6, 6, 5, 5, 0))

    # Solve the MDP
    print(colored("-" * 25 + "SOLVE THE MDP" + "-" * 25, 'red'))
    time_horizon = 20
    level_rewards, level_actions = dynamicProgramSolver(mdp, time_horizon, verbose=True)

    print(colored("-" * 25 + "SINGLE GAME SIMULATION" + "-" * 25, 'red'))
    simulate_game(mdp, level_actions, verbose=True)

def solve_minotaur_infinite_horizon_with_keys():
    # Generate the map layout
    map_layout, _, final_state = build_minotaour_map(cell_reward=0)

    # Create the MDP
    mdp = MapProblemMinotaurKeys(map_layout=map_layout,
                                 start_player=(0, 0),
                                 start_minotaur=(6, 6),
                                 start_keys=True,
                                 end_cell=final_state,
                                 key_cell=(0, 7),
                                 allow_minotaur_stay=False,
                                 probability_uniform=1,
                                 discount_factor=29/30)

    # See number of states and map layout
    print(len(mdp.states()))
    print(map_layout)

    # Print the successors
    print(colored("-" * 25 + "STATES" + "-" * 25, 'red'))

    for _ in range(10):
        state_example = mdp.states()[np.random.randint(0, len(mdp.states()))]
        print(colored("-" * 50, 'red'))
        print_successors(mdp, state_example)

    print(colored("-" * 50, 'red'))
    print_successors(mdp, (6, 5, 5, 5, 0))
    print(colored("-" * 50, 'red'))
    print_successors(mdp, (6, 6, 5, 5, 0))

    # Solve the MDP
    print(colored("-" * 25 + "SOLVE THE MDP" + "-" * 25, 'red'))
    level_rewards, level_actions = valueIterationSolver(mdp,epsilon=0.01,interactive=False)

    print(colored("-" * 25 + "SINGLE GAME SIMULATION" + "-" * 25, 'red'))
    simulate_infinte_game(mdp, level_actions[1], verbose=True)


if __name__ == '__main__':
    main()