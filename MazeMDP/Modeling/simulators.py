from MazeMDP.Modeling.mdp_modeling import MDPTerminalState


def simulate_finite_game(mdp: MDPTerminalState, policy: dict, verbose=False):
    min_level = min(policy.keys())
    max_level = max(policy.keys())
    current_state = mdp.reset()
    if verbose:
        mdp.render()

    collected_reward = 0
    for time in range(min_level, max_level + 1):

        # take decision according to MDP
        decision = policy[time][current_state]

        # Use action and see next state
        next_state, reward, terminal, _, info = mdp.step(decision)
        # Collect the reward
        collected_reward += reward

        if verbose:
            print(f"{current_state} -> {next_state} : {reward}")
            mdp.render()

        current_state = next_state

    success = True if mdp.is_goal(current_state) else False
    return success, collected_reward


def simulate_infinite_game(mdp: MDPTerminalState, policy: dict, verbose: bool = False):
    current_state = mdp.reset()
    if verbose:
        mdp.render()

    collected_reward = 0
    while True:

        decision = policy[current_state]
        next_state, reward, terminal, _, info = mdp.step(decision)
        collected_reward += reward
        if verbose:
            print(f"{current_state} -> {next_state} : {reward}")
            mdp.render()

        current_state = next_state
        if terminal:
            break

    success = True if mdp.is_goal(current_state) else False
    return success, collected_reward


if __name__ == '__main__':
    # main()
    pass
