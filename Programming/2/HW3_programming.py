'''
Done by Tejas Rane - 901015176
Sumeet Shanbhag - 641020714
'''
import time
import argparse
import numpy as np

np.random.seed(42)

# Define the possible actions and their corresponding movements in the grid
ACTIONS = {
    0: {'x': -1, 'y': 0},  # Move Up
    1: {'x': -1, 'y': 1},  # Move Up-Right
    2: {'x': 0, 'y': 1},   # Move Right
    3: {'x': 1, 'y': 1},   # Move Down-Right
    4: {'x': 1, 'y': 0},   # Move Down
    5: {'x': 1, 'y': -1},  # Move Down-Left
    6: {'x': 0, 'y': -1},  # Move Left
    7: {'x': -1, 'y': -1}  # Move Up-Left
}

class GridEnvironment():
    """ Environment Class representing the Grid World. """

    def __init__(self, deterministic=True):
        # Load the gridworld from a text file and initialize start, goal, and current state
        self.env = np.loadtxt('gridworld.txt')
        self.start = (7, 25)
        self.goal = (7, 10)
        self.current_state = self.start
        self.deterministic = deterministic  # Whether the environment is deterministic or stochastic

    def _get_next_state(self, action):
        """ Calculate the next state given the current state and action.
        
        Args:
            action (int): action to be taken

        Returns:
            next_state (tuple): next state after taking the action
        """
        # Calculate the next state based on the action taken
        next_state = (self.current_state[0] + ACTIONS[action]['x'], self.current_state[1] + ACTIONS[action]['y'])

        # If the next state is an obstacle, remain in the current state
        if self.env[next_state[0], next_state[1]] == 1:
            next_state = self.current_state

        return next_state

    def _get_reward(self, next_state):
        """ Calculate the reward for moving to the next state.
        
        Args:
            next_state (tuple): next state

        Returns:
            reward (float): reward for moving to the next state
        """
        # Assign rewards based on the next state
        reward = 0.0
        if next_state == self.goal:
            reward += 100.0  # Reward for reaching the goal
        elif next_state == self.current_state:
            reward += -50.0  # Penalty for hitting an obstacle
        else:
            reward += -1.0  # Penalty for each step taken

        return reward

    def step(self, state, action):
        """ Perform a step in the environment.
        
        Args:
            state (tuple): current state
            action (int): action to be taken

        Returns:
            p (list): transition probabilities
            next_state (list): possible next states
            reward (list): rewards for the possible next states
        """
        p = []  # Transition probabilities
        next_state = []  # Possible next states
        reward = []  # Rewards for the possible next states

        if self.deterministic:
            # In a deterministic environment, the next state and reward are certain
            self.current_state = state
            next_state.append(self._get_next_state(action))
            reward.append(self._get_reward(next_state[-1]))
            p.append(1.0)
        else:
            # In a stochastic environment, there are multiple possible next states with different probabilities
            self.current_state = state
            p = [0.8, 0.1, 0.1]  # Transition probabilities for the stochastic environment
            # Define possible actions for each action in a stochastic environment
            if action == 0:
                _actions = [action, 1, 7]
            elif action == 7:
                _actions = [action, 6, 0]
            else:
                _actions = [action, action-1, action+1]

            # Calculate possible next states and rewards for each possible action
            for a in _actions:
                next_state.append(self._get_next_state(a))
                reward.append(self._get_reward(next_state[-1]))

        return p, next_state, reward

    @property
    def shape(self):
        """ Get the shape of the environment grid.
        
        Returns:
            shape (tuple): shape of the environment grid
        """
        return self.env.shape


class Agent():
    """ Agent Class representing the learning agent in the Grid World. """

    def __init__(self, env=GridEnvironment(), gamma=0.95, accuracy_threshold=1e-3):
        self.env = env  # The environment in which the agent operates
        self.gamma = gamma  # Discount factor for future rewards
        self.accuracy_threshold = accuracy_threshold  # Convergence threshold for value iteration and policy evaluation
        self.n_actions = len(ACTIONS)  # Number of possible actions

    def policy_iteration(self):
        """ Perform Policy Iteration to find the optimal policy and value function.
        
        Returns:
            v_s (np.ndarray): Optimal value function
            policy (np.ndarray): Optimal policy
        """
        start = time.time()  # Start time for performance measurement
        # Initialize value function and policy randomly
        v_s = np.random.randn(self.env.shape[0], self.env.shape[1])
        v_s[self.env.goal[0], self.env.goal[1]] = 0  # Set the value of the goal state to 0
        policy = np.random.randint(0, self.n_actions, (self.env.shape[0], self.env.shape[1]))

        while True:
            # Policy Evaluation: Evaluate the current policy until convergence
            while True:
                delta = 0  # Track the maximum change in value function
                for i in range(self.env.shape[0]):
                    for j in range(self.env.shape[1]):
                        state = (i, j)
                        if self.env.env[state] == 1:
                            continue  # Skip obstacle states
                        v = v_s[state]  # Current value of state
                        action = policy[state]  # Action taken according to current policy
                        p, next_state, reward = self.env.step(state, action)  # Get transition probabilities, next states, and rewards
                        _v = 0
                        # Update value function based on Bellman equation
                        for p, next_state, reward in zip(p, next_state, reward):
                            _v += p * (reward + self.gamma * v_s[next_state])
                        v_s[state] = _v
                        delta = max(delta, np.abs(v - v_s[state]))  # Update maximum change in value function
                if delta < self.accuracy_threshold:
                    break  # Break if the change in value function is below the threshold

            # Policy Improvement: Improve the policy based on the current value function
            stable_policy = True  # Track whether the policy is stable
            for i in range(self.env.shape[0]):
                for j in range(self.env.shape[1]):
                    state = (i, j)
                    if self.env.env[state] == 1:
                        continue  # Skip obstacle states
                    a = policy[state]  # Current action for state
                    v_temp = []  # List to store the value of each action
                    for action in range(self.n_actions):
                        p, next_state, reward = self.env.step(state, action)  # Get transition probabilities, next states, and rewards for each action
                        _v = 0
                        # Calculate the value of each action based on Bellman equation
                        for p, next_state, reward in zip(p, next_state, reward):
                            _v += p * (reward + self.gamma * v_s[next_state])
                        v_temp.append(_v)
                    policy[state] = np.argmax(v_temp)  # Update policy to the action with maximum value
                    if a != policy[state]:
                        stable_policy = False  # If policy is changed, it is not stable

            if stable_policy:
                break  # Break if the policy is stable

        print("Policy Iteration", time.time() - start)  # Print the time taken for policy iteration
        return v_s, policy  # Return the optimal value function and policy

    def value_iteration(self):
        """ Perform Value Iteration to find the optimal policy and value function.
        
        Returns:
            v_s (np.ndarray): Optimal value function
            policy (np.ndarray): Optimal policy
        """
        start = time.time()  # Start time for performance measurement
        # Initialize value function randomly
        v_s = np.random.randn(self.env.shape[0], self.env.shape[1])
        v_s[self.env.goal[0], self.env.goal[1]] = 0  # Set the value of the goal state to 0
        policy = np.zeros((self.env.shape[0], self.env.shape[1]))  # Initialize policy to zeros

        while True:
            delta = 0  # Track the maximum change in value function
            for i in range(self.env.shape[0]):
                for j in range(self.env.shape[1]):
                    state = (i, j)
                    if self.env.env[state] == 1:
                        continue  # Skip obstacle states
                    v = v_s[state]  # Current value of state
                    v_temp = []  # List to store the value of each action
                    for action in range(self.n_actions):
                        p, next_state, reward = self.env.step(state, action)  # Get transition probabilities, next states, and rewards for each action
                        _v = 0
                        # Calculate the value of each action based on Bellman equation
                        for p, next_state, reward in zip(p, next_state, reward):
                            _v += p * (reward + self.gamma * v_s[next_state])
                        v_temp.append(_v)
                    v_s[state] = np.max(v_temp)  # Update value function to the maximum value of all actions
                    policy[state] = np.argmax(v_temp)  # Update policy to the action with maximum value
                    delta = max(delta, np.abs(v - v_s[state]))  # Update maximum change in value function
            if delta < self.accuracy_threshold:
                break  # Break if the change in value function is below the threshold

        print("Value Iteration", time.time() - start)  # Print the time taken for value iteration
        return v_s, policy  # Return the optimal value function and policy

    def generalized_policy_iteration(self):
        """ Perform Generalized Policy Iteration to find the optimal policy and value function.
        
        Returns:
            v_s (np.ndarray): Optimal value function
            policy (np.ndarray): Optimal policy
        """
        start = time.time()  # Start time for performance measurement
        # Initialize value function and policy randomly
        v_s = np.random.randn(self.env.shape[0], self.env.shape[1])
        v_s[self.env.goal[0], self.env.goal[1]] = 0  # Set the value of the goal state to 0
        policy = np.random.randint(0, self.n_actions, (self.env.shape[0], self.env.shape[1]))

        while True:
            # Policy Evaluation: Evaluate the current policy once
            for i in range(self.env.shape[0]):
                for j in range(self.env.shape[1]):
                    state = (i, j)
                    if self.env.env[state] == 1:
                        continue  # Skip obstacle states
                    action = policy[state]  # Action taken according to current policy
                    p, next_state, reward = self.env.step(state, action)  # Get transition probabilities, next states, and rewards
                    _v = 0
                    # Update value function based on Bellman equation
                    for p, next_state, reward in zip(p, next_state, reward):
                        _v += p * (reward + self.gamma * v_s[next_state])
                    v_s[state] = _v

            # Policy Improvement: Improve the policy based on the current value function
            stable_policy = True  # Track whether the policy is stable
            for i in range(self.env.shape[0]):
                for j in range(self.env.shape[1]):
                    state = (i, j)
                    if self.env.env[state] == 1:
                        continue  # Skip obstacle states
                    a = policy[state]  # Current action for state
                    v_temp = []  # List to store the value of each action
                    for action in range(self.n_actions):
                        p, next_state, reward = self.env.step(state, action)  # Get transition probabilities, next states, and rewards for each action
                        _v = 0
                        # Calculate the value of each action based on Bellman equation
                        for p, next_state, reward in zip(p, next_state, reward):
                            _v += p * (reward + self.gamma * v_s[next_state])
                        v_temp.append(_v)
                    policy[state] = np.argmax(v_temp)  # Update policy to the action with maximum value
                    if a != policy[state]:
                        stable_policy = False  # If policy is changed, it is not stable

            if stable_policy:
                break  # Break if the policy is stable

        print("Generalized Policy Iteration", time.time() - start)  # Print the time taken for generalized policy iteration
        return v_s, policy  # Return the optimal value function and policy


def plot_results(v_s, policy, filename='test.png'):
    """ Plot the results including optimal policy and value function.
    
    Args:
        v_s (np.ndarray): Optimal value function
        policy (np.ndarray): Optimal policy
        filename (str): Filename to save the plot
    """
    from matplotlib import colors
    import matplotlib.pyplot as plt

    _, [ax1, ax2] = plt.subplots(2, 1)
    ax1 = plt.subplot(2, 1, 1)

    # Plotting Actions
    gridworld = np.loadtxt('gridworld.txt')
    ax1.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    X = np.arange(-0.5, gridworld.shape[1]-1, 1)
    Y = np.arange(-0.5, gridworld.shape[0]-1, 1)
    ax1.set_xticks(X)
    ax1.set_yticks(Y)
    X = np.arange(0, gridworld.shape[1]-1, 1)
    Y = np.arange(0, gridworld.shape[0]-1, 1)
    X, Y = np.meshgrid(X, Y)
    x_ang = np.zeros_like(X)
    y_ang = np.zeros_like(Y)
    for i in range(1, policy.shape[0]-1, 1):
        for j in range(1, policy.shape[1]-1, 1):
            x_ang[i, j] = ACTIONS[policy[i, j]]['y']
            y_ang[i, j] = -ACTIONS[policy[i, j]]['x']
    QP = plt.quiver(X, Y, x_ang, y_ang, scale_units='xy', scale=3)
    plt.tick_params(bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    start = (7, 25)
    goal = (7, 10)
    gridworld[start] = -10e6
    gridworld[goal] = 10e6

    # Plotting robot path
    state = start
    while True:
        action = policy[state]
        state = (state[0] + ACTIONS[action]['x'], state[1] + ACTIONS[action]['y'])
        if gridworld[state[0], state[1]] == 1:
            state = state
        if state == goal:
            break
        gridworld[state[0], state[1]] = 5000

    cmap = colors.ListedColormap(['red', 'white', 'black', 'cyan', 'green'])
    bounds = [-10e6, 0, 1, 2, 5001, 10e6]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    ax1.imshow(gridworld, cmap=cmap, norm=norm)
    plt.title('Optimal Policy')

    # Plotting Value Function
    ax2 = plt.subplot(2, 1, 2)
    gridworld = np.loadtxt('gridworld.txt')
    ax2.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    X = np.arange(-0.5, gridworld.shape[1]-1, 1)
    Y = np.arange(-0.5, gridworld.shape[0]-1, 1)
    ax2.set_xticks(X)
    ax2.set_yticks(Y)
    plt.tick_params(bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    v_s = np.flip(v_s*(1-gridworld), 0)
    plt.pcolor(v_s, edgecolors='k', linewidths=1, cmap='gray', vmin=np.min(v_s), vmax=np.max(v_s))
    plt.title('Value Function')

    # Save Plot
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    """ Main function to run the code with specified arguments. """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--deterministic_step', help='Actions are deterministic', action='store_true')
    parser.add_argument('-t', '--dp_type', help='DP Type', type=str, default='vi')
    args = parser.parse_args()

    env = GridEnvironment(deterministic=args.deterministic_step)  # Initialize the environment
    agent = Agent(env=env)  # Initialize the agent

    if args.dp_type == 'pi':
        # Policy Iteration
        v_s, policy = agent.policy_iteration()
        plot_results(v_s, policy, filename='plots/policy_iteration.png')
    elif args.dp_type == 'vi':
        # Value Iteration
        v_s, policy = agent.value_iteration()
        plot_results(v_s, policy, filename='plots/value_iteration.png')
    elif args.dp_type == 'gpi':
        # Generalized Policy Iteration
        v_s, policy = agent.generalized_policy_iteration()
        plot_results(v_s, policy, filename='plots/generalized_policy_iteration.png')
