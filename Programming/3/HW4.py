import numpy as np
import matplotlib.pyplot as plt
from random import randint, choice, random
import random

# Define states and actions
states = list(range(6))
actions = ['L', 'R', 'T']

# Initialize global variables
Policy_Pi = []
S_A = []
Q_s = {}
Returns = {}
Episode = []
State = []
New_state = []
Reward = 0
threshold = 100  # Limit to prevent infinite loops
No_ep = 500  # Total number of episodes to run

def reset_global_variables():
    global Policy_Pi, S_A, Q_s, Returns
    
    Policy_Pi = [[0, 'T'], [1, 'L'], [2, 'L'], [3, 'L'], [4, 'R'], [5, 'T']]
    S_A = [(state, action) for state in states for action in actions if (state in [0, 5] and action == 'T') or (state not in [0, 5] and action != 'T')]
    Q_s = {pair: 0 for pair in S_A}
    Returns = {pair: [] for pair in S_A}

# Function to determine the optimal policy for a state using the first-visit method
def firstvisit_func(S, epsilon=0.05):
    # Terminal states
    if S[0] in [0, 5]:
        return 'T'
    
    # Defines the right and left actions for the current state
    right = (S[0], 'R')
    left = (S[0], 'L')
    
    # Determines the direction based on Q-values of the state
    if Q_s[left] > Q_s[right]:
        direction = -1
    elif Q_s[left] < Q_s[right]:
        direction = 1
    else:
        direction = choice([-1, 1])
    
    # Counts the number of actions available for the current state
    action_count = sum(1 for action in S_A if action[0] == S[0])
    
    # Calculates the probability for exploration vs exploitation
    exploration_prob = epsilon / action_count
    exploitation_prob = 1 - epsilon + exploration_prob
    
    # Returns the direction based on the calculated probability
    if random.random() < exploitation_prob:
        return direction
    return -direction

# Function to determine the optimal policy for a state using the exploring starts method
def exploringstart_func(S):
    # Terminal states
    if S[0] == 0 or S[0] == 5:
        return 'T'
    # Define the right and left actions for the current state
    right = (S[0], 'R')
    left = (S[0], 'L')
    # Determines the action based on Q-values of the state
    if Q_s[left] > Q_s[right]:
        return 'L'
    elif Q_s[left] < Q_s[right]:
        return 'R'
    else:
        return choice(['L', 'R'])

# Main function to run the simulation
def execute_this_func(ques):
    # Initializes a matrix to store state-values for all episodes
    Vs_matrix = np.zeros((6,No_ep))
    # Loops through all episodes
    for i in range(No_ep):
        S = 0
        G = 0
        Gamma = 0.95
        th = 0
        # Starts with a random action
        First_Action = random.choice(S_A)
        Episode = []
        # Loops until a terminal state is reached
        while S != 'T':
            Episode.append(First_Action)
            for S in Episode:
                # Determines the policy based on the question (first-visit or exploring-starts)
                if ques == 'firstvisit':
                    dir_no = firstvisit_func(S)
                    if dir_no == 'T':
                        S = 'T'
                        break
                    elif dir_no == -1:
                        S[1] = 'L'
                    else:
                        S[1] = 'R'
                elif ques == 'exploringstart':
                    S[1] = exploringstart_func(S)
                    if S[1] == 'T':
                        S = 'T'
                        break
                if S == 'T':
                    break
                # Make a stochastic decision based on a random probability
                probab = randint(1, 100)
                # Determines the next state based on the action and probability
                if S[1] == 'L' and probab <= 80:
                    State = S[0] - 1
                elif S[1] == 'L' and probab > 95:
                    State = S[0] + 1
                elif S[1] == 'L' and probab > 80 and probab <= 95:
                    State = S[0]
                elif S[1] == 'R' and probab <= 80:
                    State = S[0] + 1
                elif S[1] == 'R' and probab > 95:
                    State = S[0] - 1
                elif S[1] == 'R' and probab > 80 and probab <= 95:
                    State = S[0]
                else:
                    S = 'T'
                    break
                # Updates the episode with the new state-action pair
                for p in Policy_Pi:
                    if p[0] == State:
                        New_state_Action_Pair = p
                        Episode.append(New_state_Action_Pair)
                        th += 1
                        break
                    else:
                        continue
                # Breaks if the threshold is reached to prevent infinite loops
                if th == threshold:
                    S = 'T'
                    break
            if th == threshold:
                break
            else:
                G_next = 0
                # Calculates the total reward for the episode
                for St_Action in reversed(Episode):
                    if St_Action[0] == 0:
                        Reward = 1
                    elif St_Action[0] == 5:
                        Reward = 5
                    else:
                        Reward = 0
                    G = Reward + (Gamma * G_next)
                    G_next = G
                    # Updates the returns for the state-action pair
                    for update in Returns:
                        if update == tuple(St_Action):
                            Returns[update].append(G)
                            break
                        else:
                            continue
                # Updates the Q-values based on the returns
                for Q in Q_s:
                    for r in Returns:
                        if Q == r:
                            if len(Returns[r]) == 0:
                                Q_s[Q] = sum(Returns[r])
                            else:
                                Q_s[Q] = sum(Returns[r]) / len(Returns[r])
                            break
                        else:
                            continue
            # Updates the state-values matrix for the episode
            for j in range(6):
                L = []
                M = []
                n = 0
                for q in Q_s:
                    if q[0] == j:
                        L.append({q: Q_s[q]})
                        M.append(Q_s[q])
                        n += 1
                    else:
                        continue
                Max = max(M)
                Vs_matrix[j][i] = Max
                St = 0
                t = []
                for l in L:
                    T = l.items()
                    for key, value in T:
                        t.append([key, value])
                for key, value in t:
                    if value == Max:
                        St = key
                for p in Policy_Pi:
                    if p[0] == j:
                        no = Policy_Pi.index(p)
                        Policy_Pi[no] = list(St)
                        break
                    else:
                        continue
    # Print the results
    print(" ")
    print("optimal action value function: ")
    print(Q_s)
    print(" ")
    print("optimal policy: ")
    print(Policy_Pi)
    print(" ")
    # Plot the state-values for the episode
    plot_state_value(Vs_matrix, ques)

# Function to plot the state-values for all episodes
def plot_state_value(Vs_matrix, ques):
    fig = plt.figure()
    fig.suptitle(ques)
    plt.plot(Vs_matrix[1][:], linewidth=2, color='r', label='state 1')
    plt.plot(Vs_matrix[2][:], linewidth=2, color='k', label='state 2')
    plt.plot(Vs_matrix[3][:], linewidth=2, color='m', label='state 3')
    plt.plot(Vs_matrix[4][:], linewidth=2, color='y', label='state 4')
    plt.xlabel('Number of Episodes')
    plt.ylabel('State value function')
    plt.legend()
    plt.show()

# Main execution
if __name__ == "__main__":
    for ques, method in [("Exploring Starts", "exploring-start"), ("First Visit", "first-visit")]:
        print(ques)
        reset_global_variables()
        execute_this_func(method)
