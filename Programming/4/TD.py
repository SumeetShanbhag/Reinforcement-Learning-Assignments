import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create an empty Q-table with given dimensions
def createQ_table(rows=4, cols=12):
    q_table = np.zeros((4, cols * rows))
    return q_table

# Implement the epsilon-greedy policy for selecting actions
def epsilon_greedy_policy(state, q_table, epsilon=0.1):
    decide_explore_exploit = np.random.random()
    
    if decide_explore_exploit < epsilon:  # Exploration step
        action = np.random.choice(4)  # Choose a random action
    else:  # Exploitation step
        action = np.argmax(q_table[:, state])  # Choose the action with highest Q-value
        
    return action

# Move the agent based on the chosen action
def move_agent(agent, action):
    posX, posY = agent

    if (action == 0) and posX > 0:
        posX -= 1
    if (action == 1) and posY > 0:
        posY -= 1
    if (action == 2) and posY < 11:
        posY += 1
    if (action == 3) and posX < 3:
        posX += 1

    return posX, posY

# Retrieve the state and its corresponding maximum Q-value
def get_state(agent, q_table):
    posX, posY = agent
    state = 12 * posX + posY
    state_action = q_table[:, int(state)]
    max_state_value = np.amax(state_action)
    
    return state, max_state_value

# Get the reward based on the state
def get_reward(state):
    game_end = False
    reward = -1

    if state == 47:  # Goal state
        game_end = True
        reward = 10
    if 37 <= state <= 46:  # Cliff state
        game_end = True
        reward = -100

    return reward, game_end

# Update the Q-table based on Q-learning or SARSA update rule
def update_qTable(q_table, state, action, reward, next_state_value, gamma_discount=0.9, alpha=0.5):
    update_q_value = q_table[action, state] + alpha * (reward + (gamma_discount * next_state_value) - q_table[action, state])
    q_table[action, state] = update_q_value

    return q_table

# Mark the position visited by the agent in the environment
def visited_env(agent, env):
    posY, posX = agent
    env[posY][posX] = 1
    return env

# Implement the Q-learning algorithm
def qlearning(num_episodes=500, gamma_discount=0.9, alpha=0.5, epsilon=0.1):
    reward_cache = []  # Store cumulative rewards for each episode
    step_cache = []  # Store number of steps taken in each episode
    q_table = createQ_table()  # Initialize Q-table
    agent = (3, 0)  # Agent's start position
    
    # Loop through all episodes
    for episode in range(num_episodes):
        env = np.zeros((4, 12))
        env = visited_env(agent, env)
        agent = (3, 0)  # Reset agent's position at the start of every episode
        game_end = False
        reward_cum = 0
        step_cum = 0
        
        # Continue making steps until terminal state is reached
        while not game_end:
            state, _ = get_state(agent, q_table)
            action = epsilon_greedy_policy(state, q_table)
            agent = move_agent(agent, action)
            step_cum += 1
            env = visited_env(agent, env)
            next_state, max_next_state_value = get_state(agent, q_table)
            reward, game_end = get_reward(next_state)
            reward_cum += reward
            q_table = update_qTable(q_table, state, action, reward, max_next_state_value, gamma_discount, alpha)
            state = next_state

        reward_cache.append(reward_cum)
        step_cache.append(step_cum)

    return q_table, reward_cache, step_cache

# Implement the SARSA algorithm
def sarsa(num_episodes=500, gamma_discount=0.9, alpha=0.5, epsilon=0.1):
    q_table = createQ_table()
    step_cache = []
    reward_cache = []

    for episode in range(num_episodes):
        agent = (3, 0)
        game_end = False
        reward_cum = 0
        step_cum = 0
        env = np.zeros((4, 12))
        env = visited_env(agent, env)
        state, _ = get_state(agent, q_table)
        action = epsilon_greedy_policy(state, q_table)

        while not game_end:
            agent = move_agent(agent, action)
            env = visited_env(agent, env)
            step_cum += 1
            next_state, _ = get_state(agent, q_table)
            reward, game_end = get_reward(next_state)
            reward_cum += reward
            next_action = epsilon_greedy_policy(next_state, q_table)
            next_state_value = q_table[next_action][next_state]
            q_table = update_qTable(q_table, state, action, reward, next_state_value, gamma_discount, alpha)
            state = next_state
            action = next_action

        reward_cache.append(reward_cum)
        step_cache.append(step_cum)

    return q_table, reward_cache, step_cache

# Plot the normalized cumulative rewards for Q-learning and SARSA
def plot_reward_normalized(reward_cache_qlearning, reward_cache_SARSA):
    cum_rewards_q = []
    rewards_mean_q = np.array(reward_cache_qlearning).mean()
    rewards_std_q = np.array(reward_cache_qlearning).std()
    for i in range(0, len(reward_cache_qlearning), 10):
        normalized_reward = (sum(reward_cache_qlearning[i:i+10]) - rewards_mean_q) / rewards_std_q
        cum_rewards_q.append(normalized_reward)

    cum_rewards_SARSA = []
    rewards_mean_sarsa = np.array(reward_cache_SARSA).mean()
    rewards_std_sarsa = np.array(reward_cache_SARSA).std()
    for i in range(0, len(reward_cache_SARSA), 10):
        normalized_reward = (sum(reward_cache_SARSA[i:i+10]) - rewards_mean_sarsa) / rewards_std_sarsa
        cum_rewards_SARSA.append(normalized_reward)

    plt.plot(cum_rewards_q, label="q_learning")
    plt.plot(cum_rewards_SARSA, label="SARSA")
    plt.ylabel('Sum of Rewards during Episode')
    plt.xlabel('Batches of Episodes (sample size 10)')
    plt.title("Q-Learning/SARSA Convergence of Sum of Rewards")
    plt.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0.)
    plt.show()

# Generate a heatmap for visualizing the Q-values of the Q-table
def generate_heatmap(q_table):
    data = np.mean(q_table, axis=0).reshape((4, 12))
    sns.heatmap(data, annot=True, cmap='viridis')
    plt.show()

# Main function to run the algorithms and visualization
def main():
    q_table_SARSA, reward_cache_SARSA, step_cache_SARSA = sarsa()
    q_table_qlearning, reward_cache_qlearning, step_cache_qlearning = qlearning()

    plot_reward_normalized(reward_cache_qlearning, reward_cache_SARSA)
    print("Heatmap for Q-learning")
    generate_heatmap(q_table_qlearning)
    print("Heatmap for SARSA")
    generate_heatmap(q_table_SARSA)

if __name__ == "__main__":
    main()
