from typing import Any, Tuple

import numpy as np
from termcolor import colored

from MazeMDP.Modeling.mdp_modeling import MDPTerminalState


class MapProblemMinotaurKeys(MDPTerminalState):
    r"""
    Model of a maze with a player that might have keys or not. A minotaur is trying to capture the player.
    """

    def __init__(self,
                 map_layout: np.ndarray,
                 start_player: Tuple[int, int] = (0, 0),
                 start_minotaur: Tuple[int, int] = (5, 5),
                 start_keys: bool = False,
                 end_cell: Tuple[int, int] = (5, 5),
                 key_cell: Tuple[int, int] = (0, 5),
                 allow_minotaur_stay: bool = True,
                 probability_uniform: float = 0.5,
                 discount_factor: float = 1.0,
                 render_minotaur_position: Tuple[int, int] = (6, 5),
                 time_horizon : int =30,
                 step_cost: float = -0.0001,
                 poisoned_player: float = 0):
        r"""
        Initialize the maze problem.

        Args:
            map_layout (np.ndarray): It represents the map layout. Walls are represented by ones.
            start_player (Tuple[int,int]): Starting position of player for simulations
            start_minotaur (Tuple[int,int]): Starting position of Minotaur for simulations
            start_keys (Tuple[int,int]): Starting state of the player (with our without keys)
            end_cell (Tuple[int,int]): Goal of the player
            key_cell (Tuple[int,int]): Key position
            allow_minotaur_stay (bool): Allow minotaur to do not move in one move
            probability_uniform (float): Probability with which the minotaur does a random move. Otherwise it moves towards
                the player
            discount_factor (float): Discount factor of the MDP
            render_minotaur_position (Tuple[int,int]): Position of minotaur to show when printing results
            time_horizon (float):
            step_cost (float): Cost of doing a step towards an empty cell
            poisoned_player (float): Probability of player dying because of Poison.
        """

        self._map = map_layout
        self._actions = ['up', 'dw', 'l', 'r', 's']
        self._allow_minotaur_stay = allow_minotaur_stay
        self._probability_uniform = probability_uniform
        self._discount_factor = discount_factor
        self._time_horizon = time_horizon
        self._posioned_player = poisoned_player

        # Build the start state
        self._start_state = start_player + start_minotaur + (1,) if start_keys else (0,)
        self._end_cell = end_cell
        self._key_cell = key_cell

        ## Reward constants
        self._reward_caught = 0.0
        self._reward_arrived = 0.0

        self._reward_keys = 1
        self._reward_arriving = 1
        self._step_reward = step_cost
        self._reward_being_caught = 0.0

        ## Render constant
        self.minotaur_position = render_minotaur_position

        self._define_problem()

    def _define_problem(self):

        # Build the possible states of the problem
        self._states = [(x, y, x2, y2, k) for x in range(0, self._map.shape[0]) for y in range(0, self._map.shape[1])
                        for x2 in range(0, self._map.shape[0]) for y2 in range(0, self._map.shape[1])
                        for k in range(0, 2) if self._map[(x, y)] != 1]

        # Constant with the available actions for the player
        self._actions: [] = ['up', 'dw', 'l', 'r', 's']

        self.player_state_to_action = {}
        self.minotaur_state_to_action = {}

        # Build for each state the possible actions
        for state in self._states:

            # Build the possible actions of player and minotaur for a given state
            state_actions = self._player_actions((state[0], state[1]), check_walls=True)
            minotaur_actions = self._player_actions((state[2], state[3]), check_walls=False,
                                                    allow_stay=self._allow_minotaur_stay)

            # Add the actions to the dictionary of minotaur
            self.minotaur_state_to_action[(state[2], state[3])] = minotaur_actions

            # Restrict the action space of the player (equivalent to changing transition matrix)
            if state[0] == state[2] and state[1] == state[3]:
                self.player_state_to_action[state] = ['s']
            else:
                self.player_state_to_action[state] = state_actions

    def _player_actions(self, player_state: (int, int), check_walls=True, allow_stay=True):
        """
        Return the actions of an entity in a maze for a given state. DOes not allow entity to go out of borders or to go
        inside walls, if scpecified.
        Args:
            player_state:
            check_walls:
            allow_stay:
        """
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
        ### ABSORBING STATES ###
        # Check if the minotaur has caught the player
        if state[0] == state[2] and state[1] == state[3]:
            return [(state, 1, self.reward(state, action, state))]

        # Check if the player arrived to the goal
        if (state[0], state[1]) == self._end_cell and state[4] == 1:
            return [(state, 1, self.reward(state, action, state))]

        ### TRANSIENT STATES ###
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

        #### STATE REWARDS
        # Initial state we already have been caught
        if self.is_player_caught(state):
            return self._reward_caught

        # Initial state we already arrived without being caught , having the keys
        if self.is_goal(state):
            return self._reward_arrived

        # Generally moving and the next move arrives to the key
        if (holding_keys == 0 and state_arrival[4] == 1) and not self.is_player_caught(state_arrival):
            return self._reward_keys

        #### TRNASITION REWARDS
        # if (state_arrival[0]==state_arrival[2] and state_arrival[1]==state_arrival[2]):
        #    return self._reward_being_caught

        # Reward function when moving to a good state
        if (state_arrival[0], state_arrival[1]) == self._end_cell \
                and not self.is_player_caught(state_arrival) \
                and holding_keys == 1:
            return self._reward_arriving  # 1

        # Else return the reward of moving to the next cell
        return self._step_reward

    def minotaur_succ_prob(self, state: (int, int, int, int, int)):
        # Compute uniform moves
        mino_state = (state[2], state[3])

        # If we catched the player do not move
        if state[0] == state[2] and state[1] == state[3]:
            return [(mino_state, 1)]
        mino_actions = self.minotaur_state_to_action[mino_state]
        prob = (1 / len(mino_actions)) * self._probability_uniform

        mino_succ_uni = [(self.deterministic_f(mino_state, action), prob) for action in mino_actions]

        # Compute moves towards player

        mino_succ_towards = [(self.deterministic_f(mino_state, mino_action), (1 - self._probability_uniform) * prob) for
                             mino_action, prob in self.g_delta(state)]

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
            raise ValueError(f"{state}{action} not supported")

        return next_state_player

    def delta(self, state):
        return state[0] - state[2], state[1] - state[3]

    def g_delta(self, state):

        # Compute the delta
        delta_y, delta_x = self.delta(state)

        sgn_delta_x = np.sign(delta_x)
        sgn_delta_y = np.sign(delta_y)

        # Compute the g function
        if delta_y == 0 and delta_x == 0:
            return [('s', 1)]  # if self._allow_minotaur_stay else 'r'

        if abs(delta_x) > abs(delta_y):
            # Move in the y direction
            if sgn_delta_y == 1:
                return [('dw', 1)]
            elif sgn_delta_y == -1:
                return [('up', 1)]
            else:
                return [('l', 1)] if sgn_delta_x == -1 else [('r', 1)]

        if abs(delta_x) < abs(delta_y):
            # Move in the x direction by default
            if sgn_delta_x == 1:
                return [('r', 1)]
            elif sgn_delta_x == -1:
                return [('l', 1)]
            else:
                return [('up', 1)] if sgn_delta_y == -1 else [('dw', 1)]

        if abs(delta_y) == abs(delta_x):
            to_return = []
            if sgn_delta_x == 1:
                to_return.append(('r', 1 / 2))
            elif sgn_delta_x == -1:
                to_return.append(('l', 1 / 2))

            if sgn_delta_y == 1:
                to_return.append(('dw', 1 / 2))
            elif sgn_delta_y == -1:
                to_return.append(('up', 1 / 2))

            return to_return

    def discount(self) -> float:
        return self._discount_factor

    def print_actions(self, actions: dict):
        for i in range(self._map.shape[0]):
            for j in range(self._map.shape[1]):
                tup = (i, j) + self.minotaur_position + (0,)
                if self._map[(i, j)] == 1:
                    str_to_print = "X"
                else:
                    str_to_print = actions[tup] if actions[tup] is not None else "None"

                print("{:>5}".format(str_to_print), end=" ")
            print(" ", end=colored("|", 'red'))
            for j in range(self._map.shape[1]):
                tup = (i, j) + self.minotaur_position + (1,)

                if self._map[(i, j)] == 1:
                    str_to_print = "X"
                else:
                    str_to_print = actions[tup] if actions[tup] is not None else "None"

                print("{:>5}".format(str_to_print), end=" ")
            print()

    def print_values(self, values: dict):
        for i in range(self._map.shape[0]):
            for j in range(self._map.shape[1]):
                tup = (i, j) + self.minotaur_position + (0,)
                if self._map[(i, j)] == 1:
                    str_to_print = "X"
                    print("{:5}".format(str_to_print), end=" ")
                else:
                    print("{:5.2f}".format(values[tup]), end=" ")
            print(" ", end=colored("|", 'red'))
            for j in range(self._map.shape[1]):
                tup = (i, j) + self.minotaur_position + (1,)
                if self._map[(i, j)] == 1:
                    str_to_print = "X"
                    print("{:5}".format(str_to_print), end=" ")
                else:
                    print("{:5.2f}".format(values[tup]), end=" ")

            print()

    def save_values(self, values: dict, file_name="./value_function.txt"):
        with open(file_name, "w") as f:
            for state in self.states():
                f.write(str(state) + " " + str(values[state]) + '\n')

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

    def step(self, action) -> (Any, float, bool, bool, dict):
        succ, rew, terminal, trun, d = super(MapProblemMinotaurKeys, self).step(action)
        dead = np.random.binomial(size=1, n=1, p=self._posioned_player)
        if dead == 1:
            terminal = True
        return succ, rew, terminal, trun, d

    def is_player_caught(self, state: Any) -> bool:
        return state[0] == state[2] and state[1] == state[3]

    def is_end(self, state: Any) -> bool:
        return (state[0], state[1]) == self._end_cell or (state[0] == state[2] and state[1] == state[3])

    def is_goal(self, state: Any) -> bool:
        return (state[0], state[1]) == self._end_cell and not (state[0] == state[2] and state[1] == state[3]) and state[
            4] == 1


class Simulator:
    """
    Class used to simulate episodes under a given policy.
    """

    def __init__(self, mdp: MDPTerminalState, policy: dict, episodes: int = 10000,
                 simulator_function: Any = None, simulator_arguments: Any = None):
        """
        Initialize the simulator.
        Args:
            mdp (MDPTerminalState):
            policy (dict): Dictionary representing the policy to apply. Must be compatible with the simulator function.
            episodes (int): Number of episodes to simulate
            simulator_function(Any): Callable object for using the policy with the mdp. Must receive as parameter the mdp
                and the policy.
            simulator_arguments (dict): key arguments of the simulator function
        """

        self._simulator_arguments = {} if simulator_arguments is None else simulator_arguments
        self._episodes = episodes
        self._simulator_function = simulator_function
        self._mdp = mdp
        self._policy = policy

    def simulate(self, policy=None, verbose=False):
        policy = policy if policy is not None else self._policy
        tot_successes = 0
        tot_reward = 0
        for e in range(self._episodes):
            np.random.seed(e)
            if e % 500 == 0 and verbose:
                print(colored("-" * 25 + "[SIMULATION]  {:5}".format(e) + "-" * 25, 'green'))

            # Simulate passing the arguments given
            success, collected_reward = self._simulator_function(mdp=self._mdp, policy=policy,
                                                                 **self._simulator_arguments)

            tot_successes += 1 if success else 0
            tot_reward += collected_reward

        return tot_successes / self._episodes, tot_reward / self._episodes

    def simulate_all_times(self, verbose=False):
        max_time = max(self._policy.keys())
        min_time = min(self._policy.keys())

        print("{:10} {:10} {:15} {:15}".format("Elapsed", "TimeTL", "Probability", "Reward"))

        for t in range(max_time, min_time - 1, -1):
            #  Create sliced policy
            sliced_policy = {}
            for i in range(max_time, t - 1, -1):
                sliced_policy[i] = self._policy[i]

            # Test it
            avg_succ, avg_rew = self.simulate(policy=sliced_policy, verbose=verbose)
            print("{:10} {:10} {:15f} {:15f}".format(max_time - t + 1, t, avg_succ, avg_rew))


if __name__ == '__main__':
    pass
