import numpy as np
import matplotlib.pyplot as plt

# Constants
k = 10  # Number of slot machines (or arms)
num_steps = 1000  # Number of pulls or actions taken during each simulation run
num_runs = 2000  # Number of simulation runs for averaging results
epsilon_values = [0, 0.01, 0.1]  # Different epsilon values for exploration vs exploitation
mean = 0  # Mean of the normal distribution for initial slot machine rewards
std_dev = np.sqrt(1)  # Standard deviation (sqrt of variance)

def choose_action(Q, epsilon):
    """
    Choose action based on epsilon-greedy strategy.
    
    Parameters:
    - Q: Estimated values of actions
    - epsilon: Probability for exploration
    
    Returns:
    - Action index
    """
    if np.random.rand() < epsilon:
        return np.random.randint(k)  # Exploration: choose a random action
    else:
        return np.argmax(Q)  # Exploitation: choose the action with the highest estimated reward

def simulate_bandit(epsilon):
    """
    Returns:
    - rewards: Average rewards over steps
    - optimal_actions_count: Percentage of times optimal action was chosen
    """
    # Initialize arrays for rewards and optimal action counts
    rewards = np.zeros(num_steps)
    optimal_actions_count = np.zeros(num_steps)

    for _ in range(num_runs):
        # True reward for each slot machine (unknown to the agent)
        q_star = np.random.normal(mean, std_dev, k)
        
        # Identify the slot machine with the highest reward
        optimal_action = np.argmax(q_star)

        # Initialize estimates and counts for each slot machine
        Q = np.zeros(k)
        N = np.zeros(k)

        for step in range(num_steps):
            # Choose action using epsilon-greedy strategy
            action = choose_action(Q, epsilon)
            
            # Get the reward from chosen action
            reward = np.random.normal(q_star[action], std_dev)

            # Update counts and estimated rewards
            N[action] += 1
            Q[action] += (reward - Q[action]) / N[action]

            # Record reward and if the optimal action was chosen
            rewards[step] += reward
            if action == optimal_action:
                optimal_actions_count[step] += 1

    # Average the results over all simulation runs
    rewards /= num_runs
    optimal_actions_count /= num_runs / 100  # Convert to percentage
    return rewards, optimal_actions_count

# Plotting the results
plt.figure(figsize=(12, 8))

for epsilon in epsilon_values:
    rewards, optimal_actions_count = simulate_bandit(epsilon)
    
    # Plot average rewards over time
    plt.subplot(2, 1, 1)
    plt.plot(rewards, label=f"epsilon = {epsilon}")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend()
    
    # Plot percentage of times the optimal action was chosen
    plt.subplot(2, 1, 2)
    plt.plot(optimal_actions_count, label=f"epsilon = {epsilon}")
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.legend()

plt.tight_layout()
plt.show()
