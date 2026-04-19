import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import copy

# ========== Configuration Parameters ==========
# Modify these parameters to adjust the simulation
GRID_SIZE = 20              # Grid size (n x n)
EMPTY_RATIO = 0.3           # Ratio of empty spaces (0-1)
SIMILARITY_THRESHOLD = 0.3  # Threshold of same-type neighbors (move if below this)
MAX_ITERATIONS = 100        # Maximum number of iterations
# ============================================

def generate_positions(n, empty_ratio):
    """Generate initial matrix: 0=empty, 1=type A, 2=type B"""
    matrix = np.zeros((n, n), dtype=int)
    empty = int(n * n * empty_ratio)
    occupied = n * n - empty
    if occupied <= 0:
        return None

    while occupied > 0:
        x = np.random.randint(0, n)
        y = np.random.randint(0, n)
        if matrix[x][y] == 0:
            matrix[x][y] = 1 if np.random.random() < 0.5 else 2
            occupied -= 1
    
    return matrix

def get_neighbors(matrix, x, y):
    """Get the 8 neighbors of position (x, y)"""
    n = matrix.shape[0]
    neighbors = []
    for i in range(max(0, x-1), min(n, x+2)):
        for j in range(max(0, y-1), min(n, y+2)):
            if i != x or j != y:
                neighbors.append(matrix[i][j])
    return neighbors

def calculate_similarity(matrix, x, y):
    """
    Calculate the similarity ratio (proportion of same-type neighbors)
    Only considers non-empty neighbors
    """
    if matrix[x][y] == 0:
        return None
    
    neighbors = get_neighbors(matrix, x, y)
    non_empty_neighbors = [n for n in neighbors if n != 0]
    
    if len(non_empty_neighbors) == 0:
        return 0
    
    same_type = sum(1 for n in non_empty_neighbors if n == matrix[x][y])
    return same_type / len(non_empty_neighbors)

def is_satisfied(matrix, x, y, threshold):
    """Check if person at (x,y) is satisfied (whether they need to move)"""
    if matrix[x][y] == 0:
        return True
    
    similarity = calculate_similarity(matrix, x, y)
    return similarity >= threshold

def find_empty_position(matrix):
    """Find a random empty position"""
    empty_positions = np.argwhere(matrix == 0)
    if len(empty_positions) == 0:
        return None
    return tuple(empty_positions[np.random.randint(0, len(empty_positions))])

def perform_iteration(matrix, threshold):
    """
    Perform one iteration: traverse all occupied positions,
    swap unsatisfied people with random empty positions.
    Return True if any swap occurred, False otherwise.
    """
    occupied = np.argwhere((matrix == 1) | (matrix == 2))
    swapped = False
    
    for x, y in occupied:
        if not is_satisfied(matrix, x, y, threshold):
            empty_pos = find_empty_position(matrix)
            if empty_pos is not None:
                matrix[empty_pos] = matrix[x][y]
                matrix[x][y] = 0
                swapped = True
    
    return swapped

def simulate_schelling(n, empty_ratio, threshold, max_iterations):
    """
    Run Schelling segregation model simulation.
    Return: initial_matrix, final_matrix, iterations_count
    """
    matrix = generate_positions(n, empty_ratio)
    if matrix is None:
        return None, None, 0
    
    initial_matrix = copy.deepcopy(matrix)
    iterations = 0
    
    for i in range(max_iterations):
        if not perform_iteration(matrix, threshold):
            iterations = i
            break
        iterations = i + 1
    
    return initial_matrix, matrix, iterations

def visualize_simulation(initial_matrix, final_matrix, iterations):
    """Visualize the initial and final states of the simulation"""
    colors = ['white', 'blue', 'red']
    cmap = ListedColormap(colors)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(initial_matrix, cmap=cmap, vmin=0, vmax=2)
    axes[0].set_title('Initial State', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    axes[0].grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
    
    axes[1].imshow(final_matrix, cmap=cmap, vmin=0, vmax=2)
    axes[1].set_title(f'Final State (Iterations: {iterations})', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Row')
    axes[1].grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='Empty'),
        Patch(facecolor='blue', label='Type A'),
        Patch(facecolor='red', label='Type B')
    ]
    fig.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=11)
    
    plt.suptitle(f'Schelling Segregation Model\n(Empty Ratio: {EMPTY_RATIO}, Similarity Threshold: {SIMILARITY_THRESHOLD})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def main():
    """Main function"""
    print("=" * 50)
    print("Schelling Segregation Model Simulation")
    print("=" * 50)
    print("Parameters:")
    print(f"  Grid Size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Empty Ratio: {EMPTY_RATIO}")
    print(f"  Similarity Threshold: {SIMILARITY_THRESHOLD} (move if below this)")
    print(f"  Max Iterations: {MAX_ITERATIONS}")
    print("=" * 50)
    
    initial, final, iters = simulate_schelling(GRID_SIZE, EMPTY_RATIO, 
                                               SIMILARITY_THRESHOLD, MAX_ITERATIONS)
    
    if initial is None:
        print("Error: Could not generate matrix")
        return
    
    print("Simulation completed!")
    print(f"Actual Iterations: {iters}")
    print(f"Total Population: {np.sum((initial == 1) | (initial == 2))}")
    
    satisfactions = []
    for x, y in np.argwhere((final == 1) | (final == 2)):
        sim = calculate_similarity(final, x, y)
        if sim is not None:
            satisfactions.append(sim)
    
    if satisfactions:
        avg_satisfaction = np.mean(satisfactions)
        print(f"Average Satisfaction: {avg_satisfaction:.3f}")
    
    print("\nGenerating visualization...")
    visualize_simulation(initial, final, iters)

if __name__ == "__main__":
    main()
