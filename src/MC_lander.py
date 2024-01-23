import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from .utils import *

class MC_Control:
    def __init__(self, env, space_size, action_size, gamma=1):
        self.env = env
        self.gamma = gamma
        self.space_size = space_size
        self.action_size = action_size
        self.Qvalues = np.zeros((*self.space_size, self.action_size))
        #returns is a nested list, check utils.py for more info
        self.returns = nested_list([*self.space_size, self.action_size])

    def single_episode_update(self, traj_states, traj_rew, traj_act):
        """
        Uses a single trajectory to update the Qvalues, using first-visit MC.
        """
        # keep track of visited pair (state, action)
        visited_pairs = []
       
        # calculates the returns for each step: discounted cumulative sum
        ret = discount_cumsum(traj_rew, self.gamma)

        # given the current episode, take the last pair (St, At) and go backward
        for t_step, s in enumerate(traj_states):
            # get the action taken after being in state s
            action = traj_act[t_step]

            # build the pair (St, At) = ((x,y), At)
            pair = (s, action)

            if pair not in visited_pairs:
                s = pair[0]
                action = pair[1]

                append_value_nested(self.returns, [*s, action], ret[t_step])

                visited_pairs.append(pair)


        for pair in visited_pairs:
            s = pair[0]
            action = pair[1]
            self.Qvalues[(*s,action)] = np.mean(get_value_nested(self.returns, [*s, action]))


    def train(self, n_episodes=10000, tstar=None, epsilon_0=0.2, k_epsilon=0.0):
        # new attributes
        self.performance_traj = np.zeros(n_episodes)
        self.epsilon_0 = epsilon_0
        self.k_epsilon = k_epsilon
        self.n_episodes = n_episodes

        count = 0
        self.episode_star = None #registers the episode in which t_star count is reached
        if tstar is None:
            tstar = 2.5 * n_episodes
        epsilon = epsilon_0

        for i in range(n_episodes):
            traj_states = []
            traj_rew = []
            traj_act = []
            done = False

            s, info = self.env.reset()
            s = discretize_state(s, self.space_size)
            a = get_action_epsilon_greedy(self.Qvalues, s, epsilon, self.action_size)  

            while not done:
                count += 1
                traj_states.append(s)
                traj_act.append(a)

                new_s, r, truncated, terminated, info = self.env.step(a)
                new_s = discretize_state(new_s, self.space_size)
                done = terminated or truncated              
                traj_rew.append(r)

                # Keeps track of performance for each episode
                self.performance_traj[i] += r

                # Choose new action index
                new_a = get_action_epsilon_greedy(self.Qvalues, new_s, epsilon, self.action_size)

                a = new_a
                s = new_s
                
                if count > tstar:
                    # UPDATE OF EPSILON
                    epsilon = epsilon_0 / (1.0 + k_epsilon * (count - tstar) ** 1.05)
                    if self.episode_star is None:
                        self.episode_star = i
                    
            # MC step at the end of the episode (averaging)
            self.single_episode_update(traj_states, traj_rew, traj_act)
            if i % 100 == 0:
                print("Episode ", i, " completed")
                print("count: ", count)

    def analyse(self, n_episodes=1000):
        """
        This function analyses the agent using n_episodes.
        A greedy policy is used to choose the actions.
        """

        self.performance_traj = np.zeros(n_episodes)
        for i in range(n_episodes):
            done = False
            s, info = self.env.reset()
            s = discretize_state(s, self.space_size)

            while not done:
                a = get_action_epsilon_greedy(self.Qvalues, s, 0, self.action_size)
                
                new_s, r, truncated, terminated, info = self.env.step(a)
                done = terminated or truncated
                s = discretize_state(new_s, self.space_size)
                # Keep track of rewards for one episode
                self.performance_traj[i] += r
            if i % 100 == 0:
                print("Episode ", i, " completed")
        
        #write these values into a file, in a csv like format, also reporting the parameters
        with open("data/Fixed_t/MC_results.csv", "a") as f:
            f.write(str(self.k_epsilon) + "," + 
                    str(self.epsilon_0) + "," + 
                    str(self.cumulative_mean) + "," + 
                    str(np.mean(self.performance_traj)) + "," + 
                    str(np.std(self.performance_traj)) + "," + 
                    str(np.max(self.performance_traj)) + "," + 
                    str(np.min(self.performance_traj)) + "," + 
                    str(np.median(self.performance_traj)) + "\n")
            
           
    def plot_traj(self, cumulative=True, local=False, save_img=False):
        
        title = "MC"

        plot_indexes = np.arange(0, self.n_episodes + 1, 20, dtype=int)
        plot_indexes[-1] = plot_indexes[-1] - 1

        plt.plot(plot_indexes, self.performance_traj[plot_indexes])

        #plot a vertical line at episode_star
        plt.axvline(x=self.episode_star, color='black', linestyle='--')
        plt.text(self.episode_star + 10, 0, 't*', rotation=90)  
        
        if cumulative:
            cumulative_mean = np.cumsum(self.performance_traj) / np.arange(
                1, len(self.performance_traj) + 1
            )
            plt.plot(
                plot_indexes, cumulative_mean[plot_indexes], label="Cumulative mean"
            )
            self.cumulative_mean = cumulative_mean[-1]
        if local:
            window_size = 100
            local_mean_SARSA = np.convolve(
                self.performance_traj, np.ones(window_size) / window_size, mode="valid"
            )
            plt.plot(
                plot_indexes[plot_indexes < local_mean_SARSA.shape[0]],
                local_mean_SARSA[
                    plot_indexes[plot_indexes < local_mean_SARSA.shape[0]]
                ],
                label=" Local Mean", color = 'red'
            )
        plt.xlabel("Episode")
        plt.ylabel("Episode reward")
        plt.legend()
        plt.suptitle(f"{title} control cumulative rewards")
        plt.title(f"$\epsilon_{0}$ = {self.epsilon_0}, k = {self.k_epsilon}")

        
        if save_img:
            name = "Plots/Fixed_t/MC/MC_k_epsilon" + str(self.k_epsilon) + " epsilon_0" + str(self.epsilon_0) + ".png"
            plt.savefig(name)
        plt.show()

   