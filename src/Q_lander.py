import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from .utils import *

class Q_Learning:
    def __init__(self, env, space_size, action_size, gamma=1):
        """
        Calculates optimal policy using in-policy Temporal Difference control
        Evaluates Q-value for (S,A) pairs, using one-step updates.
        """
        self.env = env
        # the discount factor
        self.gamma = gamma

        # size of system
        self.space_size = space_size  # as tuple
        self.action_size = action_size

        # where to save returns
        self.Qvalues = np.zeros((*self.space_size, self.action_size))

    def single_step_update(self, s, a, r, new_s, done):
        """
        Uses a single step to update the values, using Temporal Difference for Q values.
        Unlike sarsa, the new action new_a is not used, as it is chosen as the argmax of Q wrt s_new.
        In this case it is stored as it is needed for the implementation of Q(lambda)
        """
        if self.old_action is not None and a != self.old_action:
            self.et = np.zeros((*self.space_size, self.action_size))
        else:
            self.et *= self.gamma * self.lambda_

        self.et[(*s, a)] += 1

        if done:
            deltaQ = r + 0 - self.Qvalues[(*s, a)]
            self.Qvalues += self.lr_v * deltaQ * self.et

        else:
            maxQ_over_actions = np.max(self.Qvalues[(*new_s,)])
            a_star = np.argmax(self.Qvalues[(*new_s,)])
            self.old_action = a_star

            deltaQ = r + self.gamma * maxQ_over_actions - self.Qvalues[(*s, a)]

            self.Qvalues += self.lr_v * deltaQ * self.et

    def train(
        self,
        n_episodes=10000,
        lambda_=0,
        tstar=None,
        epsilon_0=0.2,
        k_epsilon=0,
        lr_v0=0.15,
        k_lr=0,
    ):
        """
        This function trains the agent using n_episodes.
        The default parameters use constant learning rate and epsilon (k = 0 in both cases)
        Otherwise a decaying rate is implemented after a starting point t0 (see README for more details)
        Similarly the default implements a TD(0) evaluation procedure (lambda = 0)
        """
        self.n_episodes = n_episodes

        # Add the following attributes to the class
        self.performance_traj = np.zeros(
            n_episodes
        )  # To store cumulative reward at every game

        self.et = np.zeros((*self.space_size, self.action_size))
        self.lambda_ = lambda_

        # Parameters for epsilon decay
        self.epsilon_0 = epsilon_0  # Needed to name the plots
        self.epsilon = epsilon_0  # Needed to keep track of current epsilon
        self.k_epsilon = k_epsilon

        # Parameters for learning rate decay
        self.lr_v0 = lr_v0
        self.lr_v = lr_v0
        self.k_lr = k_lr

        self.old_action = None  # Needed for Q(lamba) learning

        if tstar is None:
            tstar = 2.5 * n_episodes

        count = 0  # counter variable needed to see when to start decaying rates
        self.episode_star = None #registers the episode in which t_star count is reached
        # Run over episodes
        for i in range(n_episodes):
            done = False
            s, _ = self.env.reset()
            s = discretize_state(s, self.space_size)
            a = get_action_epsilon_greedy(self.Qvalues, s, self.epsilon, self.action_size)
            
            while not done:
                count += 1

                # Perform one "step" in the environment
                new_s, r, truncated, terminated, _ = self.env.step(a)
                new_s = discretize_state(new_s, self.space_size)
                done = truncated or terminated

                # Keep track of rewards for one episode
                self.performance_traj[i] += r

                # Determine what action the agent would choose according to optimal policy
                new_a = get_action_epsilon_greedy(self.Qvalues, new_s, self.epsilon, self.action_size)
                # Single update with (S, A, R', S')
                self.single_step_update(s, a, r, new_s, done)

                if count > tstar:
                    self.epsilon = epsilon_0 / (
                        1.0 + self.k_epsilon * (count - tstar) ** 1.05
                    )
                    self.lr_v = lr_v0 / (1 + self.k_lr * (count - tstar) ** 0.75)
                    if self.episode_star is None:
                        self.episode_star = i
                a = new_a
                s = new_s

            if i % 100 == 0:
                print("Episode ", i, " completed")
                print("count: ", count)

    
    def analyse(self, n_episodes=1000):
        """
        This function analyses the agent using n_episodes.
        A greedy policy is used to choose the actions.
        """

        # Add the following attributes to the class
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
        with open("data/Fixed_t/Q_results.csv", "a") as f:
            f.write(str(self.lambda_) + "," + 
                    str(self.k_lr) + "," + 
                    str(self.k_epsilon) + "," + 
                    str(self.epsilon_0) + "," + 
                    str(self.cumulative_mean) + "," + 
                    str(np.mean(self.performance_traj)) + "," + 
                    str(np.std(self.performance_traj)) + "," + 
                    str(np.max(self.performance_traj)) + "," + 
                    str(np.min(self.performance_traj)) + "," + 
                    str(np.median(self.performance_traj)) + "\n")
            
            

    def plot_traj(self, cumulative=True, local=False, save_img=False):
        title = "Q Learning"

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
                label=" Local Mean",
                color="red",
            )

        plt.xlabel("Episode")
        plt.ylabel("Episode reward")
        plt.legend()
        plt.suptitle(f"{title} control cumulative rewards")

        plt.title(
            f"$\epsilon_0$ = {self.epsilon_0}, $k_\epsilon$ = {self.k_epsilon}, $\\alpha_0$ = {self.lr_v0}, $k_{{\\alpha}}$ = {self.k_lr}, $\lambda$ = {self.lambda_}"
        )
        plt.tight_layout()

        if save_img:
            name = (
                "Plots/Fixed_t/Q/Q_lambda_"
                + str(self.lambda_)
                + "_k_alpha_"
                + str(self.k_lr)
                + "k_epsilon"
                + str(self.k_epsilon)
                + "epsilon_0"
                + str(self.epsilon_0)
                + ".png"
            )
            plt.savefig(name)
        plt.show()
