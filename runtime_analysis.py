"""
Runtime Analysis Module for Melody Detection Approaches.
This module measures and compares the execution time of different melodic similarity algorithms.
Key components:
1. Test sequence generation
2. Implementation of all similarity approaches
3. Runtime measurement and statistical analysis
4. Visualization of results
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import platform
import psutil
import cpuinfo

def create_ngrams(sequence, n=4, step=1):
    """
    Create n-grams from a sequence with specified overlap.
    
    Parameters:
    - sequence: Input sequence to convert to n-grams
    - n: Length of each n-gram (default: 4)
    - step: Number of elements to slide window (default: 1)
    """
    if len(sequence) < n:
        return [sequence + [0] * (n - len(sequence))]
    return [sequence[i:i + n] for i in range(0, len(sequence) - n + 1, step)]

def generate_test_sequences(size_range=(10, 100), step=10, ngram_size=4, ngram_step=1, random_seed=42):
    """
    Generate pairs of test sequences with n-grams.
    
    Parameters:
    - size_range: Tuple (min_size, max_size) for sequence lengths
    - step: Increment between sequence sizes
    - ngram_size: Size of each n-gram (default: 4)
    - ngram_step: Step size for n-gram window (default: 1)
    - random_seed: Seed for random number generation (default: 42)
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    sizes = list(range(size_range[0], size_range[1] + 1, step))
    test_cases = []
    
    for size1 in sizes:
        for size2 in sizes:
            # Generate random sequences
            seq1 = np.random.randint(0, 7, size=size1).tolist()
            seq2 = np.random.randint(0, 7, size=size2).tolist()
            
            # Convert to n-grams
            ngrams1 = create_ngrams(seq1, ngram_size, ngram_step)
            ngrams2 = create_ngrams(seq2, ngram_size, ngram_step)
            
            test_cases.append((ngrams1, ngrams2))
    
    return test_cases

def compare_sequences(seq1, seq2):
    """Compare n-gram sequences. Returns count of positions where n-grams differ."""
    return sum(1 for x, y in zip(seq1, seq2) if not np.array_equal(x, y))

def create_cost_matrix(seq1, seq2):
    """Creates a cost matrix comparing n-gram sequences."""
    m, n = len(seq1), len(seq2)
    cost_matrix = np.zeros((m, n))
    
    for i in range(m):
        for j in range(n):
            cost_matrix[i, j] = int(not np.array_equal(seq1[i], seq2[j]))
    
    return cost_matrix

def calculate_log_transform_distance(value, max_d):
    """Transform edit distance to similarity score using logarithmic scaling."""
    try:
        log_similarity = 1 - (np.log2(1 + value) / np.log2(1 + max_d))
        return max(0.0, min(1.0, log_similarity))
    except:
        return 0.0

def meldet_approach(seq1, seq2):
    """MelDet approach modified for n-gram sequences."""
    cost_matrix = np.zeros((len(seq1), len(seq2)))
    
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            cost_matrix[i, j] = int(not np.array_equal(seq1[i], seq2[j]))
    
    m, n = cost_matrix.shape
    min_len = min(m, n)
    max_shift = abs(m - n) + 1
    best_score = float('-inf')
    
    for shift in range(max_shift):
        if m <= n:
            diag = [cost_matrix[i, i + shift] for i in range(min_len)]
        else:
            diag = [cost_matrix[i + shift, i] for i in range(min_len)]
        score = np.mean(diag)
        best_score = max(best_score, score)
    
    return best_score

def hungarian_approach(seq1, seq2):
    """Hungarian approach modified for n-gram sequences."""
    cost_matrix = np.zeros((len(seq1), len(seq2)))
    
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            cost_matrix[i, j] = int(not np.array_equal(seq1[i], seq2[j]))
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return cost_matrix[row_ind, col_ind].mean()

def sumcommon_approach(seq1, seq2):
    """
    SumCommon approach 
    Similarity = |A ∩ B| / (|A| + |B| - |A ∩ B|)
    where:
    - A = set of n-grams in song1
    - B = set of n-grams in song2
    - |A ∩ B| = number of common n-grams (exact matches)
    - |A| = total n-grams in song1
    - |B| = total n-grams in song2
    """
    common = sum(1 for x in seq1 if any(np.array_equal(x, y) for y in seq2))
    denominator = len(seq1) + len(seq2) - common
    return common / denominator if denominator > 0 else 0.0

def tversky_approach(seq1, seq2):
    """
    Tversky Index Implementation for n-gram sequences
    Based on implementation from tversky_approach.py
    
    Formula: |A ∩ B| / (|A ∩ B| + α|A\B| + β|B\A|)
    where:
    - A, B are sets of n-grams from each sequence
    - |A ∩ B| is size of intersection
    - |A\B| is size of relative complement (elements in A but not in B)
    - |B\A| is size of relative complement (elements in B but not in A)
    - α, β are weights for the complements
    """
    len1, len2 = len(seq1), len(seq2)
    
    # Initialize parameters
    total_comparisons = 0
    total_matches = 0
    
    # Compare each n-gram pair and count matches
    for i in range(len1):
        for j in range(len2):
            total_comparisons += 1
            if np.array_equal(seq1[i], seq2[j]):
                total_matches += 1
    
    # Calculate probabilities
    p1 = len1 / (len1 + len2) if (len1 + len2) > 0 else 0.5
    p2 = len2 / (len1 + len2) if (len1 + len2) > 0 else 0.5
    
    # Calculate differences using complement of matches
    matches = total_matches
    differences = total_comparisons - matches
    
    # Apply Tversky formula
    denominator = matches + (p1 * differences)
    return matches / denominator if denominator > 0 else 0.0

def measure_runtime(approach_func, test_cases, iterations=50):
    """
    Runtime Measurement Function
    
    Process:
    1. For each test case:
       - Run the approach multiple times (iterations)
       - Record execution time for each run
    2. Calculate statistics:
       - Median time per test case (more stable than mean)
       - Total and average execution times
    3. Store results with sequence size information
    
    Uses perf_counter for high-precision timing
    """
    total_time = 0
    times_per_size = []
    
    for seq1, seq2 in test_cases:
        iter_times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            _ = approach_func(seq1, seq2)
            iter_times.append(time.perf_counter() - start_time)
        
        median_time = np.median(iter_times)
        total_time += median_time
        times_per_size.append((len(seq1), len(seq2), median_time))
    
    return {
        'total_time': total_time,
        'avg_time': total_time / len(test_cases),
        'times_per_size': times_per_size
    }

def plot_runtime_comparison(results):
    """
    Bar Plot Visualization
    
    Shows:
    - Average execution time for each approach
    - Times converted to milliseconds for readability
    - Includes value labels on bars
    """
    approaches = list(results.keys())
    avg_times = [results[app]['avg_time'] * 1000 for app in approaches]
    
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 14})  # Set base font size
    
    bars = plt.bar(approaches, avg_times)
    
    plt.ylabel('Average Time per Operation (ms)', fontsize=16)
    plt.title('Runtime Comparison Across Approaches', fontsize=18, pad=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}ms',
                ha='center', va='bottom',
                fontsize=16,
                fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_size_impact(results):
    """
    Scatter Plot Visualization
    
    Shows:
    - Runtime vs sequence size for each approach
    - Includes trend lines for growth analysis
    - X-axis: average length of sequence pairs
    - Y-axis: execution time in milliseconds
    """
    plt.figure(figsize=(12, 6))
    plt.rcParams.update({'font.size': 14})  # Set base font size
    
    for approach, data in results.items():
        sizes = [(s1 + s2)/2 for s1, s2, _ in data['times_per_size']]
        times = [t * 1000 for _, _, t in data['times_per_size']]
        
        plt.scatter(sizes, times, alpha=0.5, label=approach)
        
        # Add trend line for growth pattern analysis
        z = np.polyfit(sizes, times, 1)
        p = np.poly1d(z)
        plt.plot(sizes, p(sizes), '--', alpha=0.8)
    
    plt.xlabel('Average Sequence Length', fontsize=16)
    plt.ylabel('Time (ms)', fontsize=16)
    plt.title('Runtime vs Sequence Size', fontsize=18, pad=20)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def print_runtime_results(results):
    """
    Results Table Generation
    
    Displays:
    - Total execution time
    - Average time per operation
    - Minimum and maximum times
    All times converted to appropriate units (s/ms)
    """
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

def save_results_to_csv(results):
    """
    Save runtime analysis results to CSV file.
    Includes total time, average time, and size-specific metrics for each approach.
    """
    import pandas as pd
    from pathlib import Path
    
    # Prepare data for CSV
    data = []
    for approach, metrics in results.items():
        # Add overall metrics
        row = {
            'Approach': approach,
            'Total_Time_s': metrics['total_time'],
            'Avg_Time_ms': metrics['avg_time'] * 1000,
        }
        
        # Add size-specific metrics
        for size1, size2, time in metrics['times_per_size']:
            row_with_size = row.copy()
            row_with_size.update({
                'Sequence1_Size': size1,
                'Sequence2_Size': size2,
                'Time_ms': time * 1000
            })
            data.append(row_with_size)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    output_path = Path(__file__).parent / "runtime_analysis_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

def get_system_info():
    """Get detailed system hardware information."""
    try:
        cpu_info = cpuinfo.get_cpu_info()
        cpu_details = {
            'CPU_Brand': cpu_info['brand_raw'],
            'CPU_Arch': cpu_info['arch'],
            'CPU_Cores_Physical': psutil.cpu_count(logical=False),
            'CPU_Cores_Logical': psutil.cpu_count(logical=True),
            'CPU_Frequency_MHz': psutil.cpu_freq().max if psutil.cpu_freq() else 'N/A',
            'CPU_Usage_Percent': psutil.cpu_percent(interval=1)
        }
        
        ram = psutil.virtual_memory()
        ram_details = {
            'RAM_Total_GB': round(ram.total / (1024**3), 2),
            'RAM_Available_GB': round(ram.available / (1024**3), 2),
            'RAM_Usage_Percent': ram.percent
        }
        
        os_details = {
            'OS': platform.system(),
            'OS_Version': platform.version(),
            'OS_Release': platform.release(),
            'Machine': platform.machine()
        }
        
        return {**cpu_details, **ram_details, **os_details}
    except Exception as e:
        print(f"Error getting system info: {str(e)}")
        return {}

def save_summary_report(results):
    """
    Save summarized runtime analysis results with system information.
    Includes aggregated metrics for each approach.
    """
    import pandas as pd
    from pathlib import Path
    
    system_info = get_system_info()
    summary_data = []
    
    for approach, metrics in results.items():
        times = [t for _, _, t in metrics['times_per_size']]
        summary = {
            'Approach': approach,
            'Total_Runtime_s': metrics['total_time'],
            'Average_Runtime_ms': metrics['avg_time'] * 1000,
            'Min_Runtime_ms': min(times) * 1000,
            'Max_Runtime_ms': max(times) * 1000,
            'Std_Dev_Runtime_ms': np.std([t * 1000 for t in times]),
            **system_info
        }
        summary_data.append(summary)
    
    df = pd.DataFrame(summary_data)
    output_path = Path(__file__).parent / "runtime_analysis_summary.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSummary report saved to: {output_path}")

def main():
    """
    Main Execution Flow
    
    Process:
    1. Generate test sequences of varying sizes
    2. Run each similarity approach through the test cases
    3. Measure and collect runtime data
    4. Generate visualizations and statistical analysis
    5. Display results in both graphical and tabular format
    """
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
        
        print("\nSaving detailed results to CSV...")
        save_results_to_csv(results)
        
        print("\nSaving summary report to CSV...")
        save_summary_report(results)
        
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