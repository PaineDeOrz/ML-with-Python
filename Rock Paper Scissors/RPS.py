# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

import random
import numpy as np
from collections import defaultdict

def player(prev_play, 
           opponent_history=[], 
           our_history=[], 
           q_table=defaultdict(lambda: [0.0, 0.0, 0.0]),
           learning_rate=0.1,
           discount_factor=0.9,
           epsilon=0.1):
    
    # Track history
    opponent_history.append(prev_play)
    actions = ['R', 'P', 'S']
    action_to_num = {'R': 0, 'P': 1, 'S': 2}
    counter_moves = {'R': 'P', 'P': 'S', 'S': 'R'}
    
    # Clean history (remove empty moves)
    clean_opp_history = [move for move in opponent_history if move in actions]
    
    # First move
    if len(clean_opp_history) == 0:
        choice = random.choice(actions)
        our_history.append(choice)
        return choice
    
    # Create simple state (last 2-3 opponent moves)
    state_length = min(3, len(clean_opp_history))
    state = ''.join(clean_opp_history[-state_length:])
    
    # Update Q-table if we have previous experience
    if len(our_history) > 0 and len(clean_opp_history) > 1:
        # Previous state and action
        prev_state_length = min(3, len(clean_opp_history) - 1)
        prev_state = ''.join(clean_opp_history[-prev_state_length-1:-1])
        prev_action = action_to_num[our_history[-1]]
        
        # Calculate reward
        our_last = our_history[-1]
        opp_last = clean_opp_history[-1]
        
        if counter_moves[opp_last] == our_last:
            reward = 1  # We won
        elif opp_last == our_last:
            reward = 0  # Tie
        else:
            reward = -1  # We lost
        
        # Q-learning update
        current_q = q_table[prev_state][prev_action]
        max_future_q = max(q_table[state])
        new_q = current_q + learning_rate * (reward + discount_factor * max_future_q - current_q)
        q_table[prev_state][prev_action] = new_q
    
    # Choose action using epsilon-greedy
    if random.random() < epsilon:
        # Explore: random action
        action_index = random.randint(0, 2)
    else:
        # Exploit: choose best action from Q-table
        action_index = np.argmax(q_table[state])
    
    choice = actions[action_index]
    our_history.append(choice)
    return choice
