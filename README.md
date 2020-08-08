# QuadRotor

Collection of different QuadRotor environments reconfigured to be useable with openAI gym.

quad_env1: takes in thrust for each of the four propellers and outputs position, velocity, angular velocity, orientation
Reward: from IEEE quad rotor paper

quad_env2: takes in rotor speeds for each of the four propellers and outputs position and orientation
Reward: Custom trajectory path reward 

quadHover_env: for the task of drone hovering at target location
quadTraj_env: for the task of drone following trajectory path
