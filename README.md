# Robot-World---RL

This repo contains efforts to accomplish robotic tasks using RL. 

## Multi-Joint robotic arm  

The world designed for this task is simple: The effector of the
robotic arm has to reach a target. Inspired from OpenAI Gym, it has only a few methods: 

* `env.reset()` :  to reset the world
* `env.step(action)`: interaction with world  
* `env.initRender()` and `env.render()`:  visualize the actual setup

### State 

The state is composed of each of the joints positions and the vector (target.position - effector.position). It has 2(n+1) dimensions, where n is the number of joints. (2 by default). 

### Actions

You have 2\*n actions availables, where n is the number of joints. Pair actions move the concerned joint to the left, impair actions to the right (i.e: action 0 moves joint 0 to the left, action 1 moves joint 0 to the right, action 2 moves joint 1 to the left, action 3 moves joint 1 to the right...) 
Also, you can control the number of joints, and their length.


### Reward 

Tried to keep it as simple as possible. Reward is 1/distance from effector to target. +1 if target is touched, -1 if walls or ground is touched

### Learning algo

Actor critic, failing lamentably. ac.py
