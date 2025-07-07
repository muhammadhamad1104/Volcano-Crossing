import numpy as np
import matplotlib.pyplot as plt

def define_environment(rows, cols, lava_lakes, normal_views, fabulous_views):
    """
    Define the MDP environment.
    Args:
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
        lava_lakes: List of coordinates with high penalty (lava).
        normal_views: List of coordinates with neutral/low reward.
        fabulous_views: List of coordinates with high reward.
    Returns:
        rewards: 2D array of rewards for the grid.
    """
    if not all(0 <= x < rows and 0 <= y < cols for x, y in lava_lakes + normal_views + fabulous_views):
        raise ValueError("Invalid coordinates: Ensure all coordinates are within the grid dimensions.")

    rewards = np.zeros((rows, cols))

    for lake in lava_lakes:
        rewards[lake] = -50  # High penalty for lava

    for normal in normal_views:
        rewards[normal] = 2  # Neutral/low penalty

    for fabulous in fabulous_views:
        rewards[fabulous] = 20  # High reward for fabulous views

    return rewards

def move(row, col, action, rows, cols):
    """
    Simulate moving in the grid.
    Args:
        row: Current row.
        col: Current column.
        action: Action to take (0=North, 1=East, 2=South, 3=West).
        rows: Total rows in the grid.
        cols: Total columns in the grid.
    Returns:
        new_row, new_col: New position in the grid.
    """
    if action == 0 and row > 0:  # North
        return row - 1, col
    elif action == 1 and col < cols - 1:  # East
        return row, col + 1
    elif action == 2 and row < rows - 1:  # South
        return row + 1, col
    elif action == 3 and col > 0:  # West
        return row, col - 1
    return row, col

def policy_evaluation(policy, rewards, discount=0.9, theta=1e-6):
    """
    Perform policy evaluation to find state values under a given policy.
    Args:
        policy: 3D array where policy[s][a] is the probability of taking action a in state s.
        rewards: 2D array of rewards for the grid.
        discount: Discount factor for future rewards.
        theta: Convergence threshold.
    Returns:
        values: 2D array of state values.
    """
    rows, cols = rewards.shape
    values = np.zeros((rows, cols))

    while True:
        delta = 0
        new_values = np.copy(values)

        for row in range(rows):
            for col in range(cols):
                state_value = 0
                for action, action_prob in enumerate(policy[row, col]):
                    new_row, new_col = move(row, col, action, rows, cols)
                    state_value += action_prob * (rewards[new_row, new_col] + discount * values[new_row, new_col])

                delta = max(delta, abs(state_value - values[row, col]))
                new_values[row, col] = state_value

        values = new_values
        if delta < theta:
            break

    return values

def value_iteration(rewards, discount=0.9, theta=1e-6):
    """
    Perform value iteration to find the optimal policy.
    Args:
        rewards: 2D array of rewards for the grid.
        discount: Discount factor for future rewards.
        theta: Convergence threshold.
    Returns:
        optimal_values: 2D array of optimal state values.
        optimal_policy: Dictionary mapping (row, col) -> Direction.
    """
    rows, cols = rewards.shape
    values = np.zeros((rows, cols))
    directions = ["N", "E", "S", "W"]  # Corresponding to actions 0, 1, 2, 3
    optimal_policy = {}

    while True:
        delta = 0
        new_values = np.copy(values)

        for row in range(rows):
            for col in range(cols):
                action_values = []
                for action in range(4):
                    new_row, new_col = move(row, col, action, rows, cols)
                    action_values.append(rewards[new_row, new_col] + discount * values[new_row, new_col])

                best_action_value = max(action_values)
                delta = max(delta, abs(best_action_value - values[row, col]))
                new_values[row, col] = best_action_value

        values = new_values
        if delta < theta:
            break

    # Derive policy from optimal values
    for row in range(rows):
        for col in range(cols):
            action_values = []
            for action in range(4):
                new_row, new_col = move(row, col, action, rows, cols)
                action_values.append(rewards[new_row, new_col] + discount * values[new_row, new_col])

            best_action = np.argmax(action_values)
            optimal_policy[(row, col)] = directions[best_action]

    return values, optimal_policy

def visualize_policy(policy, rows, cols):
    """
    Visualize the policy on the grid using matplotlib.
    Args:
        policy: Optimal policy dictionary mapping (row, col) -> Direction.
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
    """
    grid = np.empty((rows, cols), dtype=str)
    for (row, col), direction in policy.items():
        grid[row, col] = direction

    plt.figure(figsize=(10, 6))
    for row in range(rows):
        for col in range(cols):
            plt.text(col, rows - row - 1, grid[row, col],
                     ha='center', va='center', fontsize=12, color='black')

    plt.xticks(range(cols))
    plt.yticks(range(rows))
    plt.grid(True)
    plt.title("Optimal Policy")
    plt.show()

# Example Usage
grid_rows, grid_cols = 4, 5
lava = [(1, 2), (3, 4)]
normal = [(0, 1), (2, 2), (3, 1)]
fabulous = [(0, 4), (3, 0)]

# Part (a): Define the problem as MDP
rewards_grid = define_environment(grid_rows, grid_cols, lava, normal, fabulous)

# Part (b): Assume random policy and evaluate it
random_policy = np.ones((grid_rows, grid_cols, 4)) / 4  # Uniform policy
evaluated_values = policy_evaluation(random_policy, rewards_grid)
print("State Values under Random Policy:")
print(evaluated_values)

# Part (c): Find the optimal policy using value iteration
optimal_values, optimal_policy = value_iteration(rewards_grid)
print("\nOptimal State Values:")
print(optimal_values)

print("\nOptimal Policy (with directions):")
for row in range(grid_rows):
    for col in range(grid_cols):
        print(f"({row}, {col}) {optimal_policy[(row, col)]}", end="\t")
    print()

print("\nOptimal Policy (with directions):")
visualize_policy(optimal_policy, grid_rows, grid_cols)
