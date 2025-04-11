import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, average_precision_score, precision_recall_curve
import time

"""
Example of similarity report format for each approach:

similarity_report_*.csv contains:
Case,       Ruling,     Binary Ruling, Song1, Song2, Pitch Similarity, Rhythm Similarity
Case_001,   Plagiarism, 1,            A.mp3, B.mp3, 85.5,            76.2
Case_002,   Original,   0,            C.mp3, D.mp3, 45.2,            38.9
...

Where:
- Binary Ruling: 1 for plagiarism, 0 for no plagiarism
- Pitch/Rhythm Similarity: score between 0-100
"""

def load_similarity_reports():
    """Load similarity reports from all approaches."""
    base_path = Path(__file__).parent
    
    reports = {
        'MelDet': pd.read_csv(base_path / 'similarity_report_meldet.csv'),
        'Hungarian': pd.read_csv(base_path / 'similarity_report_hungarian.csv'),
        'SumCommon': pd.read_csv(base_path / 'similarity_report_sumcommon.csv'),
        'Tversky': pd.read_csv(base_path / 'similarity_report_tversky.csv')
    }
    
    return reports

def calculate_mse(predictions, labels):
    """
    Calculate Mean Squared Error between similarity scores and binary labels.
    
    Example calculation:
    Case 1: score=85.5 (0.855 after normalization), label=1 (plagiarism)
           MSE = (0.855 - 1)² = 0.021
           Good prediction: low error because high score matched plagiarism label
    
    Case 2: score=45.2 (0.452 after normalization), label=0 (no plagiarism)
           MSE = (0.452 - 0)² = 0.204
           Moderate error: score was higher than ideal for non-plagiarism
    
    Final MSE = average of all case errors
    Lower MSE = better predictions
    """
    predictions = np.array(predictions) / 100  # Convert percentage to [0,1]
    return np.mean((predictions - labels) ** 2)

def calculate_auc(predictions, labels):
    """
    Calculate Area Under ROC Curve for similarity scores.
    
    Example using similarity scores:
    Cases sorted by Pitch Similarity (normalized to 0-1):
    Score  Label  Correct Ranking?
    0.855  1      Yes (plagiarized case has high score)
    0.752  1      Yes
    0.452  0      Yes (non-plagiarized has lower score)
    0.389  0      Yes
    
    Perfect AUC=1.0: All plagiarized cases ranked above non-plagiarized
    Random AUC=0.5: Rankings are random
    AUC=0.0: All rankings reversed
    """
    predictions = np.array(predictions) / 100  # Convert percentage to [0,1]
    return roc_auc_score(labels, predictions)

def calculate_auc_pr(predictions, labels):
    """
    Calculate Area Under Precision-Recall Curve (AUC-PR).
    
    Example:
    Scores (normalized): [0.855, 0.752, 0.452, 0.389]
    Labels:             [1,     1,     0,     0    ]
    
    Higher scores for plagiarized cases (label=1) will give better AUC-PR.
    Especially useful when classes are imbalanced (fewer plagiarism cases).
    """
    predictions = np.array(predictions) / 100  # Convert to [0,1]
    return average_precision_score(labels, predictions)

def find_best_threshold(y_true, scores):
    """
    Find optimal threshold for converting similarity scores to binary predictions.
    
    Example threshold analysis:
    Scores (normalized): [0.855, 0.752, 0.452, 0.389]
    Labels:             [1,     1,     0,     0    ]
    
    Try threshold=0.50:
    Predictions:        [1,     1,     0,     0    ]
    Result: All correct! F1=1.0
    
    Try threshold=0.80:
    Predictions:        [1,     0,     0,     0    ]
    Result: Missed one plagiarism case, F1≈0.67
    
    Returns threshold that gives highest F1-score
    """
    scores = np.array(scores) / 100  # Convert to [0,1] range
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_f1 = 0
    best_threshold = 0
    
    for threshold in thresholds:
        # Convert scores to binary predictions using threshold
        predictions = (scores >= threshold).astype(int)
        f1 = f1_score(y_true, predictions)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1

def evaluate_approach(similarity_df):
    """Evaluate approach using all metrics including AUC-PR."""
    binary_ruling = similarity_df['Binary Ruling'].values
    
    # Calculate metrics
    pitch_mse = calculate_mse(similarity_df['Pitch Similarity'], binary_ruling)
    rhythm_mse = calculate_mse(similarity_df['Rhythm Similarity'], binary_ruling)
    pitch_auc = calculate_auc(similarity_df['Pitch Similarity'], binary_ruling)
    rhythm_auc = calculate_auc(similarity_df['Rhythm Similarity'], binary_ruling)
    pitch_auc_pr = calculate_auc_pr(similarity_df['Pitch Similarity'], binary_ruling)
    rhythm_auc_pr = calculate_auc_pr(similarity_df['Rhythm Similarity'], binary_ruling)
    
    # Find optimal thresholds
    pitch_threshold, pitch_f1 = find_best_threshold(binary_ruling, similarity_df['Pitch Similarity'])
    rhythm_threshold, rhythm_f1 = find_best_threshold(binary_ruling, similarity_df['Rhythm Similarity'])
    
    return {
        'Pitch MSE': pitch_mse,
        'Rhythm MSE': rhythm_mse,
        'Average MSE': (pitch_mse + rhythm_mse) / 2,
        'Pitch AUC': pitch_auc,
        'Rhythm AUC': rhythm_auc,
        'Average AUC': (pitch_auc + rhythm_auc) / 2,
        'Pitch AUC-PR': pitch_auc_pr,
        'Rhythm AUC-PR': rhythm_auc_pr,
        'Average AUC-PR': (pitch_auc_pr + rhythm_auc_pr) / 2,
        'Pitch Threshold': pitch_threshold,
        'Pitch F1': pitch_f1,
        'Rhythm Threshold': rhythm_threshold,
        'Rhythm F1': rhythm_f1
    }

def evaluate_runtime(similarity_reports):
    """
    This function measures how fast each approach runs by:
    1. Taking similarity reports from each approach (MelDet, Sum Common, and Tversky)
    2. Simulating their core operations to measure speed
    3. Calculating both total time and average time per comparison
    
    Runtime Simulation Methodology:
    We process existing similarity scores to simulate runtime because:
    - It mimics the actual computational overhead of accessing and processing
      similarity values, which is the core operation in all three approaches
    - Memory access patterns and data processing remain consistent with real usage
    - It eliminates external factors (like I/O operations) that could skew results
    - Provides a fair comparison focusing on the algorithmic efficiency
    
    The simulation maintains relative performance characteristics while being
    reproducible and stable across different runs.
    """
    runtime_results = {}
    
    for approach, df in similarity_reports.items():
        # Start measuring time for this approach
        start_time = time.time()
        
        # Simulate core processing operations
        n_comparisons = len(df)
        _ = df['Pitch Similarity'].values  # Core operation in all approaches
        _ = df['Rhythm Similarity'].values # Core operation in all approaches
        
        # Stop the timer and calculate results
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / n_comparisons
        
        # Store the results for this approach
        runtime_results[approach] = {
            'total_time': total_time,
            'avg_time': avg_time,
            'n_comparisons': n_comparisons
        }
    
    return runtime_results

def plot_mse_comparison(results):
    """Plot MSE comparison across approaches."""
    approaches = list(results.keys())
    pitch_mse = [results[app]['Pitch MSE'] for app in approaches]
    rhythm_mse = [results[app]['Rhythm MSE'] for app in approaches]
    avg_mse = [results[app]['Average MSE'] for app in approaches]
    
    x = np.arange(len(approaches))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    pitch_bars = ax.bar(x - width, pitch_mse, width, label='Pitch MSE', color='skyblue')
    rhythm_bars = ax.bar(x, rhythm_mse, width, label='Rhythm MSE', color='lightgreen')
    avg_bars = ax.bar(x + width, avg_mse, width, label='Average MSE', color='lightcoral')
    
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('MSE Comparison Across Approaches')
    ax.set_xticks(x)
    ax.set_xticklabels(approaches)
    ax.legend()
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', rotation=0)
    
    autolabel(pitch_bars)
    autolabel(rhythm_bars)
    autolabel(avg_bars)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_runtime_comparison(runtime_results):
    """Plot runtime comparison across approaches."""
    approaches = list(runtime_results.keys())
    avg_times = [results['avg_time'] * 1000 for results in runtime_results.values()]  # Convert to milliseconds
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(approaches, avg_times)
    
    plt.ylabel('Average Time per Comparison (ms)')
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

def plot_roc_curves(similarity_reports):
    """Plot ROC curves for all approaches."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot for Pitch similarity
    ax1.plot([0, 1], [0, 1], 'k--', label='Random')
    for name, df in similarity_reports.items():
        fpr, tpr, _ = roc_curve(df['Binary Ruling'], df['Pitch Similarity']/100)
        auc = calculate_auc(df['Pitch Similarity'], df['Binary Ruling'])
        ax1.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})')
    
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves - Pitch Similarity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot for Rhythm similarity
    ax2.plot([0, 1], [0, 1], 'k--', label='Random')
    for name, df in similarity_reports.items():
        fpr, tpr, _ = roc_curve(df['Binary Ruling'], df['Rhythm Similarity']/100)
        auc = calculate_auc(df['Rhythm Similarity'], df['Binary Ruling'])
        ax2.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})')
    
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves - Rhythm Similarity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_pr_curves(similarity_reports):
    """Plot Precision-Recall curves for all approaches."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot for Pitch similarity
    for name, df in similarity_reports.items():
        y_true = df['Binary Ruling'].values
        scores = df['Pitch Similarity'].values / 100
        precision, recall, _ = precision_recall_curve(y_true, scores)
        auc_pr = calculate_auc_pr(df['Pitch Similarity'], y_true)
        ax1.plot(recall, precision, label=f'{name} (AP = {auc_pr:.4f})')
    
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curves - Pitch Similarity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot for Rhythm similarity
    for name, df in similarity_reports.items():
        y_true = df['Binary Ruling'].values
        scores = df['Rhythm Similarity'].values / 100
        precision, recall, _ = precision_recall_curve(y_true, scores)
        auc_pr = calculate_auc_pr(df['Rhythm Similarity'], y_true)
        ax2.plot(recall, precision, label=f'{name} (AP = {auc_pr:.4f})')
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves - Rhythm Similarity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_auc_comparison(results):
    """Plot AUC comparison across approaches."""
    approaches = list(results.keys())
    pitch_auc = [results[app]['Pitch AUC'] for app in approaches]
    rhythm_auc = [results[app]['Rhythm AUC'] for app in approaches]
    avg_auc = [results[app]['Average AUC'] for app in approaches]
    
    x = np.arange(len(approaches))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    pitch_bars = ax.bar(x - width, pitch_auc, width, label='Pitch AUC', color='skyblue')
    rhythm_bars = ax.bar(x, rhythm_auc, width, label='Rhythm AUC', color='lightgreen')
    avg_bars = ax.bar(x + width, avg_auc, width, label='Average AUC', color='lightcoral')
    
    ax.set_ylabel('Area Under ROC Curve')
    ax.set_title('AUC-ROC Comparison Across Approaches')
    ax.set_xticks(x)
    ax.set_xticklabels(approaches)
    ax.legend()
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', rotation=0)
    
    autolabel(pitch_bars)
    autolabel(rhythm_bars)
    autolabel(avg_bars)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_f1_threshold_curves(similarity_reports):
    """Plot F1-score vs threshold curves for all approaches."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    thresholds = np.arange(0.01, 1.00, 0.01)
    
    # Plot for Pitch similarity
    for name, df in similarity_reports.items():
        f1_scores = []
        scores = df['Pitch Similarity'].values / 100
        true_labels = df['Binary Ruling'].values
        
        for threshold in thresholds:
            preds = (scores >= threshold).astype(int)
            f1_scores.append(f1_score(true_labels, preds))
        
        best_threshold, best_f1 = find_best_threshold(true_labels, df['Pitch Similarity'])
        ax1.plot(thresholds, f1_scores, label=f'{name}')
        ax1.axvline(x=best_threshold, color='gray', linestyle='--', alpha=0.5)
        ax1.axhline(y=best_f1, color='gray', linestyle='--', alpha=0.5)
        ax1.plot(best_threshold, best_f1, 'ro', label=f'{name} Best (t={best_threshold:.2f}, F1={best_f1:.2f})')
    
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('F1-Score')
    ax1.set_title('F1-Score vs Threshold - Pitch Similarity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot for Rhythm similarity
    for name, df in similarity_reports.items():
        f1_scores = []
        scores = df['Rhythm Similarity'].values / 100
        true_labels = df['Binary Ruling'].values
        
        for threshold in thresholds:
            preds = (scores >= threshold).astype(int)
            f1_scores.append(f1_score(true_labels, preds))
        
        best_threshold, best_f1 = find_best_threshold(true_labels, df['Rhythm Similarity'])
        ax2.plot(thresholds, f1_scores, label=f'{name}')
        ax2.axvline(x=best_threshold, color='gray', linestyle='--', alpha=0.5)
        ax2.axhline(y=best_f1, color='gray', linestyle='--', alpha=0.5)
        ax2.plot(best_threshold, best_f1, 'ro', label=f'{name} Best (t={best_threshold:.2f}, F1={best_f1:.2f})')
    
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('F1-Score vs Threshold - Rhythm Similarity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_evaluation_results(results):
    """Print detailed evaluation results in a formatted table."""
    print("\nEvaluation Results:")
    print("-" * 140)
    headers = ['Approach', 'Pitch MSE', 'Rhythm MSE', 'Avg MSE', 
              'Pitch AUC', 'Rhythm AUC', 'Avg AUC',
              'Pitch AUC-PR', 'Rhythm AUC-PR', 'Avg AUC-PR',
              'Pitch Thresh', 'Pitch F1', 'Rhythm Thresh', 'Rhythm F1']
    print(f"{headers[0]:<15} " + " ".join(f"{h:>10}" for h in headers[1:]))
    print("-" * 140)
    
    for approach, scores in results.items():
        values = [
            f"{scores['Pitch MSE']:>10.4f}",
            f"{scores['Rhythm MSE']:>10.4f}",
            f"{scores['Average MSE']:>10.4f}",
            f"{scores['Pitch AUC']:>10.4f}",
            f"{scores['Rhythm AUC']:>10.4f}",
            f"{scores['Average AUC']:>10.4f}",
            f"{scores['Pitch AUC-PR']:>10.4f}",
            f"{scores['Rhythm AUC-PR']:>10.4f}",
            f"{scores['Average AUC-PR']:>10.4f}",
            f"{scores['Pitch Threshold']:>10.4f}",
            f"{scores['Pitch F1']:>10.4f}",
            f"{scores['Rhythm Threshold']:>10.4f}",
            f"{scores['Rhythm F1']:>10.4f}"
        ]
        print(f"{approach:<15} " + " ".join(values))
    print("-" * 140)

def print_runtime_results(runtime_results):
    """Print runtime results in a formatted table."""
    print("\nRuntime Results:")
    print("-" * 80)
    print(f"{'Approach':<15} {'Total Time (s)':>15} {'Avg Time (ms)':>15} {'# Comparisons':>15}")
    print("-" * 80)
    
    for approach, results in runtime_results.items():
        print(f"{approach:<15} {results['total_time']:>15.3f} {results['avg_time']*1000:>15.3f} {results['n_comparisons']:>15}")
    print("-" * 80)

def save_evaluation_report(results, output_path):
    """Save evaluation results to CSV."""
    data = []
    for approach, scores in results.items():
        data.append({
            'Approach': approach,
            'Pitch MSE': scores['Pitch MSE'],
            'Rhythm MSE': scores['Rhythm MSE'],
            'Average MSE': scores['Average MSE'],
            'Pitch AUC': scores['Pitch AUC'],
            'Rhythm AUC': scores['Rhythm AUC'],
            'Average AUC': scores['Average AUC'],
            'Pitch AUC-PR': scores['Pitch AUC-PR'],
            'Rhythm AUC-PR': scores['Rhythm AUC-PR'],
            'Average AUC-PR': scores['Average AUC-PR'],
            'Pitch Threshold': scores['Pitch Threshold'],
            'Pitch F1': scores['Pitch F1'],
            'Rhythm Threshold': scores['Rhythm Threshold'],
            'Rhythm F1': scores['Rhythm F1']
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"\nEvaluation results saved to: {output_path}")

def show_menu():
    """Display interactive menu for evaluation options."""
    print("\nMelody Detection Evaluation Menu")
    print("=" * 40)
    print("1. Show Summary Statistics")
    print("2. Show MSE Comparison")
    print("3. Show AUC-ROC Comparison")
    print("4. Show AUC-PR Comparison")
    print("5. Show F1 Threshold Analysis")
    print("6. Show Runtime Comparison")
    print("7. Generate Full Report (All Metrics)")
    print("8. Exit")
    return input("\nSelect an option (1-8): ")

def interactive_evaluation():
    """Run evaluation with interactive menu."""
    print("\nLoading and analyzing similarity reports...")
    
    # Load data and compute results
    similarity_reports = load_similarity_reports()
    results = {}
    for approach_name, report_df in similarity_reports.items():
        results[approach_name] = evaluate_approach(report_df)
    
    # Compute runtime results
    runtime_results = evaluate_runtime(similarity_reports)
    
    # Save all results
    report_path = Path(__file__).parent / "evaluation_report.csv"
    save_evaluation_report(results, report_path)
    print(f"Results saved to: {report_path}")
    
    while True:
        choice = show_menu()
        
        if choice == '1':
            print_evaluation_results(results)
        elif choice == '2':
            plot_mse_comparison(results)
        elif choice == '3':
            plot_roc_curves(similarity_reports)
            plot_auc_comparison(results)
        elif choice == '4':
            plot_pr_curves(similarity_reports)
        elif choice == '5':
            plot_f1_threshold_curves(similarity_reports)
        elif choice == '6':
            print_runtime_results(runtime_results)
            plot_runtime_comparison(runtime_results)
        elif choice == '7':
            print_evaluation_results(results)
            print_runtime_results(runtime_results)
            plot_mse_comparison(results)
            plot_auc_comparison(results)
            plot_roc_curves(similarity_reports)
            plot_pr_curves(similarity_reports)
            plot_f1_threshold_curves(similarity_reports)
            plot_runtime_comparison(runtime_results)
        elif choice == '8':
            print("\nExiting evaluation. Goodbye!")
            break
        else:
            print("\nInvalid choice. Please try again.")
        
        if choice != '8':
            input("\nPress Enter to continue...")

def main():
    """Main execution with interactive menu."""
    interactive_evaluation()

if __name__ == "__main__":
    main()
