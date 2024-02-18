import numpy as np
import matplotlib.pyplot as plt

class DynaMaze:
    # Initialization of the maze, including the start, goal, obstacles, and Q-table
    def __init__(self, maze, start_state, goal_state, obstacles, n_planning_steps=5, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.maze = np.array(maze)
        self.start_state = start_state
        self.goal_state = goal_state
        self.obstacles = obstacles
        self.n_planning_steps = n_planning_steps
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        
        self.Q = dict()  # Initialize Q-values
        self.model = dict()  # Initialize the model
        self.initialize_Q_and_model()

    def initialize_Q_and_model(self):
        for row in range(self.maze.shape[0]):
            for col in range(self.maze.shape[1]):
                for action in self.actions:
                    self.Q[((row, col), action)] = 0.0
                    self.model[((row, col), action)] = (0.0, (row, col))  # Initial model predicts no reward and no movement

    # Method to choose an action based on epsilon-greedy policy    
    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return self.actions[np.random.choice(len(self.actions))]  # Explore
        else:
            q_values = [self.Q[(state, a)] for a in self.actions]
            max_q = max(q_values)
            # If multiple actions have the same max Q-value, select randomly among them
            actions_with_max_q = [self.actions[i] for i, q in enumerate(q_values) if q == max_q]
            return actions_with_max_q[np.random.choice(len(actions_with_max_q))]  # Exploit

    # Method to take action and observe the next state and reward
    def take_action(self, state, action):
        next_state = (state[0] + action[0], state[1] + action[1])
        # If next state is an obstacle or out of bounds, stay in the same state
        if next_state in self.obstacles or not (0 <= next_state[0] < self.maze.shape[0] and 0 <= next_state[1] < self.maze.shape[1]):
            next_state = state
        reward = 1 if next_state == self.goal_state else 0
        return reward, next_state
    
    def update_Q_value(self, state, action, reward, next_state):
        max_q_next_state = max(self.Q[(next_state, a)] for a in self.actions)
        self.Q[(state, action)] += self.alpha * (reward + self.gamma * max_q_next_state - self.Q[(state, action)])
    
    def update_model(self, state, action, reward, next_state):
        self.model[(state, action)] = (reward, next_state)

    def planning(self):
        for _ in range(self.n_planning_steps):
            # Randomly select a previously observed state and action
            state, action = list(self.model.keys())[np.random.choice(len(self.model))]
            reward, next_state = self.model[(state, action)]
            # Update Q-value with model predictions
            self.update_Q_value(state, action, reward, next_state)
    
    def run_episode(self):
        state = self.start_state
        steps = 0
        while state != self.goal_state:
            action = self.choose_action(state)
            reward, next_state = self.take_action(state, action)
            self.update_Q_value(state, action, reward, next_state)
            self.update_model(state, action, reward, next_state)
            self.planning()
            state = next_state
            steps += 1
            if steps > 10000:  # Adding a safety break to prevent infinite loops
                break
        return steps

def simulate_dyna_maze(n_planning_steps, num_episodes=50):
    dyna_maze = DynaMaze(maze, start_state, goal_state, obstacles, n_planning_steps=n_planning_steps)
    steps_per_episode = []
    for episode in range(num_episodes):
        steps = dyna_maze.run_episode()
        steps_per_episode.append(steps)
    return steps_per_episode

# Define the maze dimensions and the structure of the maze
maze_height, maze_width = 6, 9
start_state = (2, 0)
goal_state = (0, 8)
obstacles = [(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)]

# Initialize the maze with zeros
maze = np.zeros((maze_height, maze_width))

# Mark the start and goal state in the maze
maze[start_state] = -1  # -1 for start state
maze[goal_state] = 1    # 1 for goal state

# Mark obstacles in the maze
for obs in obstacles:
    maze[obs] = 2  # 2 for obstacles

# Planning step values to simulate
planning_steps_values = [0, 5, 50]
learning_curves = {}

# Simulate for each n_planning_steps value and record the learning curves
for n in planning_steps_values:
    learning_curves[n] = simulate_dyna_maze(n_planning_steps=n)

# Plot the learning curves
plt.figure(figsize=(12, 8))
for n, steps in learning_curves.items():
    plt.plot(steps, label=f'n={n}')
plt.xlabel('Episodes')
plt.ylabel('Steps per episode')
plt.title('Learning Curves for Different Planning Steps in Dyna-Q')
plt.legend()
plt.grid()
plt.show()