"""
Runtime Analysis Module for Melody Detection Approaches.
Measures actual execution time of core algorithms.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

def generate_test_sequences(size_range=(10, 100), step=10):
    """Generate test sequences of different sizes."""
    sizes = list(range(size_range[0], size_range[1] + 1, step))
    test_cases = []
    
    for size1 in sizes:
        for size2 in sizes:
            # Generate random sequences and convert to Python lists
            seq1 = np.random.randint(0, 7, size=size1).tolist()
            seq2 = np.random.randint(0, 7, size=size2).tolist()
            test_cases.append((seq1, seq2))
    
    return test_cases

def compare_sequences(seq1, seq2):
    """Compare individual n-gram sequences and count differences."""
    return sum(1 for x, y in zip(seq1, seq2) if x != y)

def create_cost_matrix(seq1, seq2):
    """Calculate edit distance matrix between two sequences."""
    m, n = len(seq1), len(seq2)
    cost_matrix = np.zeros((m, n))
    
    for i in range(m):
        for j in range(n):
            cost_matrix[i, j] = int(seq1[i] != seq2[j])
    
    return cost_matrix

def calculate_log_transform_distance(value, max_d):
    """Transform edit distance to similarity score using logarithmic scaling."""
    try:
        log_similarity = 1 - (np.log2(1 + value) / np.log2(1 + max_d))
        return max(0.0, min(1.0, log_similarity))
    except:
        return 0.0

def meldet_approach(seq1, seq2):
    """
    MelDet Approach - Diagonal-based similarity
    
    Steps:
    1. Create cost matrix where cell[i,j] = 1 if elements are different, 0 if same
    2. For each possible diagonal (|m-n|+1 diagonals):
       - Extract diagonal values
       - Calculate average score
    3. Return the best (minimum) diagonal score
    
    Note: Only considers diagonals of length min(m,n) to ensure valid alignments
    """
    # Create cost matrix
    cost_matrix = np.zeros((len(seq1), len(seq2)))
    
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            cost_matrix[i, j] = int(seq1[i] != seq2[j])
    
    # Find best diagonal
    m, n = cost_matrix.shape
    min_len = min(m, n)
    max_shift = abs(m - n) + 1
    scores = []
    
    for shift in range(max_shift):
        if m <= n:
            diag = [cost_matrix[i, i + shift] for i in range(min_len)]
        else:
            diag = [cost_matrix[i + shift, i] for i in range(min_len)]
        scores.append(np.mean(diag))
    
    return min(scores)

def hungarian_approach(seq1, seq2):
    """
    Hungarian Approach - Optimal assignment
    
    Steps:
    1. Create cost matrix where cell[i,j] = 1 if elements are different, 0 if same
    2. Apply Hungarian algorithm to find minimum cost assignment
       - Finds optimal pairing between elements of seq1 and seq2
       - Each element is paired exactly once
    3. Return average cost of optimal assignment
    
    Note: Guarantees globally optimal matching but doesn't preserve order
    """
    # Create cost matrix
    cost_matrix = np.zeros((len(seq1), len(seq2)))
    
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            cost_matrix[i, j] = int(seq1[i] != seq2[j])
    
    # Apply Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return cost_matrix[row_ind, col_ind].mean()

def sumcommon_approach(seq1, seq2):
    """
    SumCommon Approach - Set-based similarity
    
    Formula: |A ∩ B| / (|A| + |B| - |A ∩ B|)
    where:
    - |A ∩ B| is number of common elements
    - |A| + |B| is sum of sequence lengths
    - Subtracting |A ∩ B| avoids counting common elements twice
    """
    # Make copy of sequences to avoid modifying originals
    seq1_copy = seq1.copy()
    seq2_copy = seq2.copy()
    
    # Calculate intersection (common elements)
    common = sum(1 for x in seq1_copy if x in seq2_copy)
    
    # Calculate total elements minus duplicates
    denominator = len(seq1) + len(seq2) - common
    
    return common / denominator if denominator > 0 else 0.0

def tversky_approach(seq1, seq2):
    """
    Tversky Approach - Set-based similarity with position

    Steps:
    1. Calculate weights based on sequence lengths:
       alpha = len(seq1) / (len(seq1) + len(seq2))
       beta = len(seq2) / (len(seq1) + len(seq2))
    2. Find common elements and differences
    3. Calculate Tversky index:
       score = common / (common + α*diff1 + β*diff2)
    
    Note: Weights reflect relative sequence lengths
    """
    # Calculate weights based on sequence lengths
    len1, len2 = len(seq1), len(seq2)
    total_len = len1 + len2
    alpha = len1 / total_len if total_len > 0 else 0.5
    beta = len2 / total_len if total_len > 0 else 0.5
    
    # Create sets while preserving position information
    set1 = [(i, x) for i, x in enumerate(seq1)]
    set2 = [(i, x) for i, x in enumerate(seq2)]
    
    # Find common elements with their positions
    common = sum(1 for i1, x1 in set1 for i2, x2 in set2 if x1 == x2)
    
    # Calculate differences considering positions
    diff1 = sum(1 for i1, x1 in set1 if not any(x1 == x2 for _, x2 in set2))
    diff2 = sum(1 for i2, x2 in set2 if not any(x2 == x1 for _, x1 in set1))
    
    denominator = common + (alpha * diff1) + (beta * diff2)
    return common / denominator if denominator > 0 else 0.0

def measure_runtime(approach_func, test_cases, iterations=50):
    """Measure runtime for a specific approach."""
    total_time = 0
    times_per_size = []
    
    for seq1, seq2 in test_cases:
        iter_times = []
        for _ in range(iterations):
            # Ensure time.time() has enough precision
            start_time = time.perf_counter()
            _ = approach_func(seq1, seq2)
            iter_times.append(time.perf_counter() - start_time)
        
        # Use median time for stability
        median_time = np.median(iter_times)
        total_time += median_time
        times_per_size.append((len(seq1), len(seq2), median_time))
    
    return {
        'total_time': total_time,
        'avg_time': total_time / len(test_cases),
        'times_per_size': times_per_size
    }

def plot_runtime_comparison(results):
    """Plot runtime comparison across approaches."""
    approaches = list(results.keys())
    avg_times = [results[app]['avg_time'] * 1000 for app in approaches]  # Convert to milliseconds
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(approaches, avg_times)
    
    plt.ylabel('Average Time per Operation (ms)')
    plt.title('Runtime Comparison Across Approaches')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}ms',
                ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_size_impact(results):
    """Plot impact of sequence size on runtime."""
    plt.figure(figsize=(12, 6))
    
    for approach, data in results.items():
        sizes = [(s1 + s2)/2 for s1, s2, _ in data['times_per_size']]  # Average sequence length
        times = [t * 1000 for _, _, t in data['times_per_size']]  # Convert to milliseconds
        
        plt.scatter(sizes, times, alpha=0.5, label=approach)
        
        # Add trend line
        z = np.polyfit(sizes, times, 1)
        p = np.poly1d(z)
        plt.plot(sizes, p(sizes), '--', alpha=0.8)
    
    plt.xlabel('Average Sequence Length')
    plt.ylabel('Time (ms)')
    plt.title('Runtime vs Sequence Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def print_runtime_results(results):
    """Print runtime results in a formatted table."""
    print("\nRuntime Analysis Results:")
    print("-" * 80)
    headers = ['Approach', 'Total Time (s)', 'Avg Time (ms)', 'Min Size Time (ms)', 'Max Size Time (ms)']
    print(f"{headers[0]:<15} {headers[1]:>15} {headers[2]:>15} {headers[3]:>15} {headers[4]:>15}")
    print("-" * 80)
    
    for approach, data in results.items():
        times = [t for _, _, t in data['times_per_size']]
        print(f"{approach:<15} "
              f"{data['total_time']:>15.3f} "
              f"{data['avg_time']*1000:>15.3f} "
              f"{min(times)*1000:>15.3f} "
              f"{max(times)*1000:>15.3f}")
    print("-" * 80)

def main():
    """Main execution for runtime analysis."""
    print("Starting runtime analysis...")
    try:
        print("\nGenerating test sequences...")
        test_cases = generate_test_sequences(size_range=(10, 100), step=10)
        
        print(f"Running analysis on {len(test_cases)} test cases...")
        approaches = {
            'MelDet': meldet_approach,
            'Hungarian': hungarian_approach,
            'SumCommon': sumcommon_approach,
            'Tversky': tversky_approach
        }
        
        results = {}
        for name, func in approaches.items():
            print(f"\nAnalyzing {name} approach...")
            results[name] = measure_runtime(func, test_cases)
        
        print("\nGenerating results...")
        print_runtime_results(results)
        
        print("\nGenerating comparison plots...")
        plot_runtime_comparison(results)
        plot_size_impact(results)
        
    except Exception as e:
        print(f"\nError during runtime analysis: {str(e)}")
        return
    
    print("\nRuntime analysis completed successfully.")
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()