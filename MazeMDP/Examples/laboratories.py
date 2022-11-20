import time

from termcolor import colored
import numpy as np

from MazeMDP.Modeling.maze_modeling_keys import MapProblemMinotaurKeys, Simulator
from MazeMDP.Modeling.mdp_modeling import print_successors,save_average_reward
from MazeMDP.Maps.mapbuilders import build_minotaour_map
from MazeMDP.Modeling.simulators import simulate_finite_game, simulate_infinite_game
from MazeMDP.Modeling.solvers import valueIterationSolver, dynamicProgramSolver


def solve_minotaur_finite_horizon_with_keys():
    # Generate the map layout
    map_layout, _, final_state = build_minotaour_map(cell_reward=0)

    # Create the MDP
    mdp = MapProblemMinotaurKeys(map_layout=map_layout,
                                 start_player=(0, 0),
                                 start_minotaur=(6, 5),
                                 start_keys=True,
                                 end_cell=final_state,
                                 key_cell=(0, 7),
                                 allow_minotaur_stay=True,
                                 probability_uniform=1,
                                 step_cost=-0.0001)

    # See number of states and map layout
    print(len(mdp.states()))
    print(map_layout)

    save_average_reward(mdp)

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

    # 1.d
    # Solve the MDP
    print(colored("-" * 25 + "SOLVE THE MDP" + "-" * 25, 'red'))
    time_horizon = 30
    level_rewards, level_actions = dynamicProgramSolver(mdp, time_horizon, verbose=True)

    # Show a single game for 1.c
    print(colored("-" * 25 + "SINGLE GAME SIMULATION" + "-" * 25, 'red'))
    simulate_finite_game(mdp, level_actions, verbose=True)

    # Simulate multiple games for t=1,30 for 1.d
    print(colored("-" * 25 + "MULTIPLE GAME SIMULATION" + "-" * 25, 'red'))

    sim = Simulator(mdp, policy=level_actions, episodes=10000, simulator_function=simulate_finite_game)
    avg_succ,avg_rew = sim.simulate(verbose=True)
    print("Average success is {:5f}".format(avg_succ))
    print("Average reward is {:5f}".format(avg_rew))


    print(colored("-" * 25 + "MULTIPLE TIME HORIZONS" + "-" * 25, 'red'))

    sim.simulate_all_times()


def solve_minotaur_infinite_horizon_with_keys():
    # Generate the map layout
    map_layout, _, final_state = build_minotaour_map(cell_reward=0)

    # Create the MDP
    mdp = MapProblemMinotaurKeys(map_layout=map_layout,
                                 start_player=(0, 0),
                                 start_minotaur=(6, 5),
                                 start_keys=True,
                                 end_cell=final_state,
                                 key_cell=(0, 7),
                                 allow_minotaur_stay=False,
                                 probability_uniform=0.65,
                                 discount_factor=29 / 30,
                                 step_cost=0,
                                 poisoned_player=1/30)

    # See number of states and map layout
    print(len(mdp.states()))
    print(map_layout)

    save_average_reward(mdp)

    # Print the successors
    print(colored("-" * 25 + "STATES" + "-" * 25, 'red'))

    for _ in range(10):
        state_example = mdp.states()[np.random.randint(0, len(mdp.states()))]
        print(colored("-" * 50, 'red'))
        print_successors(mdp, state_example)


    for state in mdp.states():
        # Look only states with keys
        if state[4] == 1:
            actions = mdp.actions(state)
            for action in actions:
                successors = mdp.successor_prob_reward(state,action)

    print(colored("-" * 50, 'red'))
    print_successors(mdp, (6, 5, 5, 5, 0))
    print(colored("-" * 50, 'red'))
    print_successors(mdp, (6, 6, 5, 5, 0))

    # Solve the MDP
    print(colored("-" * 25 + "SOLVE THE MDP" + "-" * 25, 'red'))
    level_rewards, level_actions = valueIterationSolver(mdp, epsilon=0.01, interactive=False)

    # Print the results
    mdp.print_values(level_rewards[1])
    mdp.print_actions(level_actions[1])
    mdp.save_values(level_rewards[1])

    # Simulate
    print(colored("-" * 25 + "SINGLE GAME SIMULATION" + "-" * 25, 'red'))
    simulate_infinite_game(mdp, level_actions[1], verbose=True)




    ## Simulate to compute probabilities
    print(colored("-" * 25 + "MULTIPLE GAME SIMULATION" + "-" * 25, 'red'))
    episodes = 10000
    tot_successes = 0
    tot_reward = 0
    for e in range(episodes):
        if e % 500 == 0:
            print(colored("-" * 25 + "[SIMULATION]  {:5}".format(e) + "-" * 25, 'green'))
        success, collected_reward = simulate_infinite_game(mdp, level_actions[1])

        tot_successes += 1 if success else 0
        tot_reward += collected_reward

    print("Average success is {:5f}".format(tot_successes / episodes))
    print("Average reward is {:5f}".format(tot_reward / episodes))


def main():
    t = time.time()
    # 1.c
    solve_minotaur_finite_horizon_with_keys()
    print(colored("ELAPSED TIME : {:5f} seconds".format(time.time() - t),'red'))
    t = time.time()
    # 1.d
    solve_minotaur_infinite_horizon_with_keys()
    print(colored("ELAPSED TIME : {:5f} seconds".format(time.time() - t),'red'))

if __name__ == '__main__':
    main()