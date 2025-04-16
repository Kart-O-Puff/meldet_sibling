import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, average_precision_score, precision_recall_curve

"""
Example of similarity report format for each approach:

similarity_report_*.csv contains:
Case,         Ruling,      Binary Ruling,    Song1,  Song2,   Pitch Similarity, Rhythm Similarity
Case_001,     Plagiarism,        1,          A.mid,  B.mid,         85.5,              76.2
Case_002,     No Plagiarism,     0,          C.mid,  D.mid,         45.2,              38.9
...

Where:
- Binary Ruling: 1 for plagiarism, 0 for no plagiarism
- Pitch/Rhythm Similarity: score between 0-100 (%)
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
    Calculate Area Under Precision-Recall Curve (AUC-PR), also known as Average Precision (AP).
    
    The Average Precision (AP) is calculated by sklearn.metrics.average_precision_score as:
    AP = Σ (R_n - R_n-1) * P_n
    
    Where:
    - P_n = precision at threshold n
    - R_n = recall at threshold n
    - The thresholds are automatically determined by unique prediction scores
    
    Input values:
    - predictions: similarity scores (0-100) from either Pitch or Rhythm comparison
                  these are normalized to 0-1 range by dividing by 100
    - labels: Binary Ruling column from similarity report (1=plagiarism, 0=no plagiarism)
    
    Example calculation with similarity scores [85, 75, 45, 35] and labels [1, 1, 0, 0]:
    1. Normalize scores: [0.85, 0.75, 0.45, 0.35]
    2. Sort by decreasing score and track corresponding labels:
       Score: 0.85  Label: 1  Precision: 1/1=1.00  Recall: 1/2=0.50  ΔRecall: 0.50
       Score: 0.75  Label: 1  Precision: 2/2=1.00  Recall: 2/2=1.00  ΔRecall: 0.50
       Score: 0.45  Label: 0  Precision: 2/3=0.67  Recall: 2/2=1.00  ΔRecall: 0.00
       Score: 0.35  Label: 0  Precision: 2/4=0.50  Recall: 2/2=1.00  ΔRecall: 0.00
    
    3. AP = (1.00 * 0.50) + (1.00 * 0.50) = 1.00
       Perfect AP because all plagiarized cases had higher scores than non-plagiarized
    """
    predictions = np.array(predictions) / 100  # Convert percentage to [0,1]
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
        ap = average_precision_score(y_true, scores)
        ax1.plot(recall, precision, label=f'{name} (AP = {ap:.4f})')
    
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curves - Pitch Similarity')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    
    # Plot for Rhythm similarity
    for name, df in similarity_reports.items():
        y_true = df['Binary Ruling'].values
        scores = df['Rhythm Similarity'].values / 100
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax2.plot(recall, precision, label=f'{name} (AP = {ap:.4f})')
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves - Rhythm Similarity')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
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

def display_pr_analysis_table(similarity_df, feature='Pitch Similarity'):
    """
    Display how different similarity thresholds affect plagiarism detection decisions.
    Shows what happens when we decide "similarity scores above X% mean plagiarism"
    """
    scores = similarity_df[feature].values / 100
    y_true = similarity_df['Binary Ruling'].values
    thresholds = np.arange(0.0, 1.1, 0.1)
    
    results = []
    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))  # Correctly identified plagiarism
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))  # False plagiarism alerts
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))  # Missed plagiarism cases
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))  # Correctly identified non-plagiarism

        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        results.append({
            'Threshold': threshold,
            'True Positives': tp,
            'False Positives': fp,
            'False Negatives': fn,
            'True Negatives': tn,
            'Precision': precision,
            'Recall': recall
        })
    
    print(f"\nPlagiarism Detection Analysis for {feature}")
    print("=" * 85)
    print("If we consider scores above X as plagiarism:")
    print("-" * 85)
    print(f"{'X (threshold)':>15} {'Detected':>15} {'Detected as NOT':>15} {'Missed':>15} {'Accuracy of ':>15} {'Coverage of Real ':>15}")
    print(f"{'(as %)':>15} {'Plagiarism (Correct)':>20} {'Plagiarism':>15} {'Actual Plagiarism':>15} {'Detection':>15} {'Plagiarism Cases':>15}")
    print("-" * 85)
    
    for row in results:
        threshold_pct = row['Threshold'] * 100
        print(f"{threshold_pct:>15.1f}% {row['True Positives']:>12d} {row['False Positives']:>12d} {row['False Negatives']:>12d} {row['Precision']:>12.4f} {row['Recall']:>12.4f}")
    print("-" * 85)
    print("Precision Rate = Detected Plagiarism (Correct) / Total Detections (Accuracy of Plagiarism Detection)")
    print("Recall Rate = Detected Plagiarism (Correct) / Total Actual Plagiarism Cases")

def print_evaluation_results(results, show_pr_analysis=False):
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
    
    if show_pr_analysis:
        # Add precision-recall analysis tables with consistent formatting
        similarity_reports = load_similarity_reports()
        pd.set_option('display.float_format', lambda x: '%.4f' % x)
        
        for approach, df in similarity_reports.items():
            print(f"\n{approach} Analysis:")
            display_pr_analysis_table(df, 'Pitch Similarity')
            display_pr_analysis_table(df, 'Rhythm Similarity')
        
        pd.reset_option('display.float_format')

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
    print("1. Show Basic Statistics")
    print("2. Show MSE Comparison")
    print("3. Show AUC-ROC Comparison")
    print("4. Show AUC-PR Comparison")
    print("5. Show Precision-Recall Analysis")
    print("6. Show F1 Threshold Analysis")
    print("7. Generate Full Report (All Metrics)")
    print("8. Exit")
    return input("\nSelect an option (1-8): ")

def show_approach_menu(approaches):
    """Display menu for selecting an approach and feature."""
    print("\nSelect an analysis option:")
    print("=" * 30)
    for i, approach in enumerate(approaches, 1):
        print(f"{i}. {approach} - Pitch Analysis")
        print(f"{i+len(approaches)}. {approach} - Rhythm Analysis")
    next_num = 2 * len(approaches) + 1
    print(f"{next_num}. All Approaches - Pitch Analysis")
    print(f"{next_num+1}. All Approaches - Rhythm Analysis")
    print(f"{next_num+2}. All Approaches - Both Analyses")
    print("0. Back to main menu")
    return input(f"\nSelect an option (0-{next_num+2}): ")

def interactive_evaluation():
    """Run evaluation with interactive menu."""
    print("\nLoading and analyzing similarity reports...")
    
    # Load data and compute results
    similarity_reports = load_similarity_reports()
    results = {}
    for approach_name, report_df in similarity_reports.items():
        results[approach_name] = evaluate_approach(report_df)
    
    # Save all results
    report_path = Path(__file__).parent / "evaluation_report.csv"
    save_evaluation_report(results, report_path)
    print(f"Results saved to: {report_path}")
    
    while True:
        choice = show_menu()
        
        if choice == '1':
            # Basic Statistics without PR Analysis
            print("\nBasic Evaluation Results:")
            print("-" * 110)
            headers = ['Approach', 'Pitch MSE', 'Rhythm MSE', 'Avg MSE', 
                      'Pitch AUC-ROC', 'Rhythm AUC-ROC', 'Avg AUC-ROC']
            print(f"{headers[0]:<15} " + " ".join(f"{h:>12}" for h in headers[1:]))
            print("-" * 110)
            
            for approach, scores in results.items():
                values = [
                    f"{scores['Pitch MSE']:>12.4f}",
                    f"{scores['Rhythm MSE']:>12.4f}",
                    f"{scores['Average MSE']:>12.4f}",
                    f"{scores['Pitch AUC']:>12.4f}",
                    f"{scores['Rhythm AUC']:>12.4f}",
                    f"{scores['Average AUC']:>12.4f}"
                ]
                print(f"{approach:<15} " + " ".join(values))
            print("-" * 110)
            input("\nPress Enter to continue...")
        elif choice == '2':
            plot_mse_comparison(results)
        elif choice == '3':
            plot_roc_curves(similarity_reports)
            plot_auc_comparison(results)
        elif choice == '4':
            plot_pr_curves(similarity_reports)
        elif choice == '5':
            # Approach selection submenu for PR Analysis
            approaches = list(similarity_reports.keys())
            n_approaches = len(approaches)
            while True:
                approach_choice = show_approach_menu(approaches)
                if approach_choice == '0':
                    break
                elif approach_choice.isdigit():
                    choice_num = int(approach_choice)
                    if 1 <= choice_num <= n_approaches:
                        # Pitch analysis for specific approach
                        approach = approaches[choice_num - 1]
                        print(f"\nPitch Analysis for {approach}:")
                        display_pr_analysis_table(similarity_reports[approach], 'Pitch Similarity')
                    elif n_approaches < choice_num <= 2 * n_approaches:
                        # Rhythm analysis for specific approach
                        approach = approaches[choice_num - n_approaches - 1]
                        print(f"\nRhythm Analysis for {approach}:")
                        display_pr_analysis_table(similarity_reports[approach], 'Rhythm Similarity')
                    elif choice_num == 2 * n_approaches + 1:
                        # All approaches - Pitch
                        print("\nPitch Analysis for All Approaches:")
                        for approach, df in similarity_reports.items():
                            print(f"\n{approach}:")
                            display_pr_analysis_table(df, 'Pitch Similarity')
                    elif choice_num == 2 * n_approaches + 2:
                        # All approaches - Rhythm
                        print("\nRhythm Analysis for All Approaches:")
                        for approach, df in similarity_reports.items():
                            print(f"\n{approach}:")
                            display_pr_analysis_table(df, 'Rhythm Similarity')
                    elif choice_num == 2 * n_approaches + 3:
                        # All approaches - Both
                        print("\nComplete Analysis for All Approaches:")
                        for approach, df in similarity_reports.items():
                            print(f"\n{approach}:")
                            display_pr_analysis_table(df, 'Pitch Similarity')
                            display_pr_analysis_table(df, 'Rhythm Similarity')
                    else:
                        print("\nInvalid choice. Please try again.")
                else:
                    print("\nInvalid choice. Please try again.")
                
                if approach_choice != '0':
                    input("\nPress Enter to see approach menu again...")
        
        elif choice == '6':
            plot_f1_threshold_curves(similarity_reports)
        elif choice == '7':
            print_evaluation_results(results, show_pr_analysis=True)
            plot_mse_comparison(results)
            plot_auc_comparison(results)
            plot_roc_curves(similarity_reports)
            plot_pr_curves(similarity_reports)
            plot_f1_threshold_curves(similarity_reports)
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
