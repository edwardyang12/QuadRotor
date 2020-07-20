# QuadRotor

Collection of different QuadRotor environments reconfigured to be useable with openAI gym.

quad_env1: takes in thrust for each of the four propellers and outputs position, velocity, angular velocity, orientation
Reward: from IEEE quad rotor paper

quad_env2: takes in rotor speeds for each of the four propellers and outputs position and orientation
Reward: Custom trajectory path reward 

quad_env3: takes in force and intertial moments and outputs position, acceleration, quaternion, angular acceleration
Reward: Uses a non- ML trajectory controller

