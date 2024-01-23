# Lunar-Lander

This repository contains the final project for Reinforcement Learning course of the Master Degree in Data Science and Scientific Computing, A.Y. 2022-2023. The slides for the final presentation are available [here](Lunar_Lander_presentation.pdf).

Goal of this project is to solve the [LunarLander-v2](https://gymnasium.farama.org/environments/box2d/lunar_lander/) OpenAI Gymnasium environment using various model-free reinforcement learning algorithms, namely:

- [Monte Carlo Control](src/MC_lander.py)
- [SARSA](src/SARSA_lander.py)
- [Expected SARSA](src/ESARSA_lander.py)
- [Q-Learning](src/Q_lander.py)

Each model's hyperparameters are tuned using a grid search approach. The results of this analysis are available in the following folders:

- [data](data) folder, for a general statistical analysis of the performance.
- [Plots](Plots) folder, for a visual representation of the trajectories. 

Final considerations are reported in the presentation.


## Training Clips

These training snapshots are captured using a greedy policy after the training phase (~10000 episodes). A random agent is also
provided for comparison:  

**Random** 
 <img src="gifs/random.gif" width="200" height="300"/>

**Monte-Carlo**  
<img src="gifs/MC.gif" width="200" height="300"/>

**SARSA**
<img src="gifs/SARSA.gif" width="200" height="300"/>

**Expected SARSA**
<img src="gifs/ESARSA.gif" width="200" height="300"/>

**Q-learning**  
<img src="gifs/Q.gif" width="200" height="300"/>

## Implementation References  

1. Tutor's notebooks [Emanuele Panizon](https://www.ictp.it/member/emanuele-panizon)
2. [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
3. [Solving The Lunar Lander Problem under Uncertainty using Reinforcement Learning](https://arxiv.org/abs/2011.11850)

