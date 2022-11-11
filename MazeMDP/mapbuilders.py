import numpy as np


WALL_REWARD = 1


# Functions for building the map

def add_vertical_wall(map: np.ndarray, column: int, start: int, end: int,wall_value = WALL_REWARD):
    map[start:(end + 1), column] = wall_value


def add_horizontal_wall(map: np.ndarray, row: int, start: int, end: int,wall_value = WALL_REWARD):
    map[row, start:(end + 1)] = wall_value



def build_simple_map():
    final_state = (5,5)
    map_dimension = (6,7)

    # Build the map layout
    map_layout = np.zeros(map_dimension)
    add_vertical_wall(map_layout, 2, 0, 2)
    add_horizontal_wall(map_layout, 4, 1, 5)

    # Build the map rewards
    map_rewards = np.ones(map_dimension) * -0.1
    map_rewards[final_state] = 20
    add_vertical_wall(map_rewards, 2, 0, 2,-30)
    add_horizontal_wall(map_rewards, 4, 1, 5,-30)

    # Build the MDP
    return map_layout,map_rewards,(5,5)



def build_blueberries_map():
    final_state = (5, 5)
    map_dimension = (6, 7)
    blueberrymap = [[0.0, 1.0, -float("inf"), 10.0, 10.0, 10.0, 10.0],
                    [0.0, 1.0, float("-inf"), 10.0, 0.0, 0.0, 10.0],
                    [0.0, 1.0, float("-inf"), 10.0, 0.0, 0.0, 10.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 10.0],
                    [0.0, float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), 10],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 11.0, 10.0]]
    map_rewards = np.array(blueberrymap)
    # Build the map layout
    map_layout = np.zeros(map_dimension)
    add_vertical_wall(map_layout, 2, 0, 2)
    add_horizontal_wall(map_layout, 4, 1, 5)
    return map_layout, map_rewards, (5, 5)



def build_minotaour_map():

    final_state = (6,5)
    map_dimension = (7,8)


    map_layout = np.zeros(map_dimension)
    add_vertical_wall(map_layout,2,0,3)
    add_horizontal_wall(map_layout,5,1,6)
    add_horizontal_wall(map_layout,2,5,7)
    map_layout[(1,5)] = WALL_REWARD
    map_layout[(3,5)] = WALL_REWARD
    map_layout[(6,4)] = WALL_REWARD

    # Build the map rewards
    map_rewards = np.ones(map_dimension) * -0.1
    map_rewards[final_state] = 20
    add_vertical_wall(map_rewards, 2, 0, 3,-30)
    add_horizontal_wall(map_rewards, 5, 1, 6,-30)
    add_horizontal_wall(map_rewards, 2, 5, 7,-30)
    map_rewards[(1, 5)] = -30
    map_rewards[(3, 5)] = -30
    map_rewards[(6, 4)] = -30

    return map_layout, map_rewards, final_state















