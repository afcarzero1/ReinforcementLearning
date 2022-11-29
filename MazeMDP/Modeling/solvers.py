import os

from termcolor import colored

from MazeMDP.Modeling.mdp_modeling import MDP


def dynamicProgramSolver(problem: MDP, simulation_time: int = 5,verbose=False):
    """
    Solves the given MDP problem using dynamic programming. Suited for solving problems having a finite horizon.
    Args:
        problem (MDP) : Markov Decision Process to solve
        simulation_time (int) : It represents the simulation time horizon
        verbose (bool) : Display full information
    """
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
                maximum_reward_state = float("-inf")
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
                maximum_reward_state = float("-inf")
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

        # Print for having an idea of what is happening
        if verbose:
            problem.print_values(remaining_reward)
            problem.print_actions(best_action)

    return level_rewards, level_actions


def valueIterationSolver(problem: MDP, epsilon: float = 1e-10, interactive: bool = False):
    """
    Solve the MDP using value iteration.
    Args:
        problem (MDP) : Markov Decision Process to solve
        epsilon (float) : Error tolerance for finishing the iterations
        interactive (boolean): Solve one iteration at a time and continue on user input
    Returns:
        remaining_reward(dict) : Dictionary containing the value function for each state
        policy (dict) : Dictionary containing the best action for each state.
    """
    remaining_reward = {}  # Dictionary (state -> average remaining reward at a given level)

    def Q(state, action):
        su = sum(prob * (reward + problem.discount() * remaining_reward[newState]) \
                 for newState, prob, reward in problem.successor_prob_reward(state, action))
        return su

    # Initialize the rewards
    for state in problem.states():
        remaining_reward[state] = 0

    max_iterations = 1000000
    T = max_iterations
    for level in range(T, 0, -1):
        # Apply the recursion algorithm
        next_remaining_reward = {}  # (state -> remaining reward at level)
        for state in problem.states():
            # todo : add here support for mdp with terminal state
            # if isinstance(problem,MDPTerminalState):

            # We have the maximum of the Q function
            next_remaining_reward[state] = max(Q(state, action) for action in problem.actions(state))

        # Compute the difference for stopping conditions
        if max(abs(remaining_reward[state] - next_remaining_reward[state]) for state in problem.states()) < epsilon:
            break

        # Update the remaining values
        remaining_reward = next_remaining_reward.copy()
        if level % 10 == 0:
            print(colored("[ITERATION] {:5}/ {:10}".format(T-level,T),'green'))
        # Read policy if interactive
        if interactive:
            policy = {}
            for state in problem.states():
                # todo : add also here mdoification for mdp with terminal state

                actions = [(action,Q(state,action)) for action in problem.actions(state)]
                act,val = max(actions, key=lambda x : x[1])
                policy[state] = act


            os.system('clear')
            problem.print_values(remaining_reward)
            problem.print_actions(policy)
            input()
    policy = {}
    for state in problem.states():
        # todo : add also here mdoification for mdp with terminal state

        actions = [(action, Q(state, action)) for action in problem.actions(state)]
        act, val = max(actions, key=lambda x: x[1])
        policy[state] = act


    return {1:remaining_reward},{1:policy}

