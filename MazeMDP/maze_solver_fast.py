import numpy as np
import colorama

WALL_REWARD = -30


def add_vertical_wall(map: np.ndarray, column: int, start: int, end: int):
    map[start:(end + 1), column] = WALL_REWARD


def add_horizontal_wall(map: np.ndarray, row: int, start: int, end: int):
    map[row, start:(end + 1)] = WALL_REWARD


def build_map() -> np.ndarray:
    dimension = (6, 7)
    map = -np.ones(dimension)

    add_vertical_wall(map, 2, 0, 2)
    add_horizontal_wall(map, 4, 1, 5)
    map[5, 5] = 20
    return map


class MapProblem():
    def __init__(self, map: np.ndarray, final_state: (int, int)):
        self.map = map
        self.final_state = final_state
        self.N = map.size

        # create x*y states
        self._states = [(x, y) for x in range(0, map.shape[0]) for y in range(0, map.shape[1])]
        self._actions = ['up', 'dw', 'l', 'r']

    def isEnd(self, state):
        return state == self.final_state

    def actions(self, state):
        actions = self._actions.copy()
        if state[0] <= 0:
            actions.remove('up')
        if state[0] >= (self.map.shape[0] - 1):
            actions.remove('dw')
        if state[1] <= 0:
            actions.remove('l')
        if state[1] >= (self.map.shape[1] - 1):
            actions.remove('r')
        if self.isEnd(state):
            actions = []

        return actions

    def succProbReward(self, state, action):

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

        reward = self.map[next_state]
        return [(next_state, prob, reward)]

    def startState(self):
        return (0, 0)

    def states(self):
        return self._states

    def discount(self):
        return 1

    def print_values(self, values: dict):

        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                tup = (i, j)
                print("{:5.2f}".format(values[tup]), end=" ")
            print()

    def print_Actions(self,actions : dict):
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                tup = (i, j)
                str_to_print = actions[tup] if actions[tup] is not None else "None"
                print("{:>5}".format(str_to_print), end=" ")
            print()



def dynamicProgramSolver(problem: MapProblem,simulation_time = 5):
    T = simulation_time

    optimal_decisions = {}  # (level,state) ->
    remaining_reward = {}
    best_action = {}

    level_rewards = {}
    level_actions = {}


    def Q(state,action):
        su = sum(prob * (reward + problem.discount() * remaining_reward[newState]) \
                                            for newState, prob, reward in problem.succProbReward(state, action))


    for level in range(T, 0, -1):
        print(f"Level {level}")

        if level == T:
            # maximize over reachable states
            for state in problem.states():
                maximum_reward_state = 0
                best_state_action = None
                # Compute the maximum reward reachable from this state
                for action in problem.actions(state):
                    # For a given action we can end up in several states with a certain probability.
                    # We choose the action that leads to the best
                    rewards = [reward for next_state, prob, reward in problem.succProbReward(state, action)]
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
            for state in problem.states():
                maximum_reward_state = 0
                prev_rewards = level_rewards[level + 1]
                best_state_action = None
                for action in problem.actions(state):
                    # Given the action we can end up in different states
                    max_reward_action = sum(prob * (reward + problem.discount() * prev_rewards[newState]) \
                                            for newState, prob, reward in problem.succProbReward(state, action))

                    # If we obtain an average reward
                    if max_reward_action > maximum_reward_state:
                        maximum_reward_state = max_reward_action
                        best_state_action = action

                remaining_reward[state] = maximum_reward_state
                best_action[state] = best_state_action

            level_rewards[level] = remaining_reward.copy()
            level_actions[level] = best_action.copy()
            # print(remaining_reward)

        problem.print_values(remaining_reward)
        problem.print_Actions(best_action)

    return level_rewards,level_actions


def printPath(problem : MapProblem,solution : dict):
    min_level = min(solution.keys())
    max_level = max(solution.keys())

    decisions = []
    state = problem.startState()
    for level in range(min_level,max_level):
        level_decisions = solution[level][state]
        decisions.append((state,level_decisions))

        succ = problem.succProbReward(state,level_decisions)


    return decisions


def main():
    mdp = MapProblem(build_map(), (5, 5))

    print("THE MAP IS :")
    print(build_map())
    # mdp = MapProblem(map)

    states = mdp.states()

    print("The states are")
    print(states)
    print("Action of state 0 is ")
    print(mdp.actions(states[0]))
    print(mdp.succProbReward(states[0], mdp.actions(states[0])[0]))

    print(mdp.succProbReward((1, 0), 'r'))

    _,level_actions = dynamicProgramSolver(mdp,20)
    decisions = printPath(mdp,level_actions)
    print(decisions)

if __name__ == '__main__':
    main()
