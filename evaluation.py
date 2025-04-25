import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, average_precision_score, precision_recall_curve
from scipy import stats
from scikit_posthocs import posthoc_nemenyi_friedman

"""
Evaluation Module for Comparing Similarity Approaches

The Friedman Test is a non-parametric statistical test used to:
1. Detect differences in treatments across multiple attempts/samples
2. Compare three or more matched or paired groups
3. Test the null hypothesis that matched samples were drawn from the same population

Key Components of Friedman Test:
- Null Hypothesis (H0): No significant difference between the approaches
- Alternative Hypothesis (H1): At least one approach differs significantly
- Significance Level: Typically α = 0.05

Friedman Test Calculation:
1. Rank the approaches within each case (1 = best/lowest MSE, 4 = worst/highest MSE)
2. Calculate the test statistic:
   Q = [12/(bk(k+1))] * [∑(Rj²)] - 3b(k+1)
   where:
   - b = number of blocks (cases)
   - k = number of approaches
   - Rj = sum of ranks for each approach
   
Interpretation:
- If p-value < 0.05: Reject H0, significant differences exist
- If p-value ≥ 0.05: Fail to reject H0, no significant differences

Post-hoc Analysis (Nemenyi test):
- Only performed if Friedman test is significant
- Compares approaches pairwise to identify which ones differ
- Critical value based on number of approaches and significance level
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
    """Calculate Mean Squared Error and Standard Deviation between similarity scores and binary labels."""
    predictions = np.array(predictions) / 100  # Convert percentage to [0,1]
    squared_errors = (predictions - labels) ** 2
    return np.mean(squared_errors), np.std(squared_errors)

def calculate_auc(predictions, labels):
    predictions = np.array(predictions) / 100  # Convert percentage to [0,1]
    return roc_auc_score(labels, predictions)

def calculate_auc_pr(predictions, labels):
    
    predictions = np.array(predictions) / 100  # Convert percentage to [0,1]
    return average_precision_score(labels, predictions)

def find_best_threshold(y_true, scores):
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
    """Evaluate approach using all metrics including squared errors."""
    binary_ruling = similarity_df['Binary Ruling'].values
    
    # Calculate squared errors for pitch and rhythm
    pitch_scores = similarity_df['Pitch Similarity'].values / 100
    rhythm_scores = similarity_df['Rhythm Similarity'].values / 100
    pitch_se = (pitch_scores - binary_ruling) ** 2
    rhythm_se = (rhythm_scores - binary_ruling) ** 2
    
    # Calculate MSE and other metrics
    pitch_mse = np.mean(pitch_se)
    rhythm_mse = np.mean(rhythm_se)
    pitch_mse_std = np.std(pitch_se)
    rhythm_mse_std = np.std(rhythm_se)
    combined_mse = (pitch_mse + rhythm_mse) / 2
    combined_mse_std = np.sqrt((pitch_mse_std**2 + rhythm_mse_std**2) / 2)
    
    pitch_auc = calculate_auc(similarity_df['Pitch Similarity'], binary_ruling)
    rhythm_auc = calculate_auc(similarity_df['Rhythm Similarity'], binary_ruling)
    pitch_auc_pr = calculate_auc_pr(similarity_df['Pitch Similarity'], binary_ruling)
    rhythm_auc_pr = calculate_auc_pr(similarity_df['Rhythm Similarity'], binary_ruling)
    
    pitch_threshold, pitch_f1 = find_best_threshold(binary_ruling, similarity_df['Pitch Similarity'])
    rhythm_threshold, rhythm_f1 = find_best_threshold(binary_ruling, similarity_df['Rhythm Similarity'])
    
    return {
        'Pitch MSE': pitch_mse,
        'Pitch MSE Std': pitch_mse_std,
        'Rhythm MSE': rhythm_mse,
        'Rhythm MSE Std': rhythm_mse_std,
        'Combined Features MSE': combined_mse,
        'Combined Features MSE Std': combined_mse_std,
        'Pitch AUC': pitch_auc,
        'Rhythm AUC': rhythm_auc,
        'Average AUC': (pitch_auc + rhythm_auc) / 2,
        'Pitch AUC-PR': pitch_auc_pr,
        'Rhythm AUC-PR': rhythm_auc_pr,
        'Average AUC-PR': (pitch_auc_pr + rhythm_auc_pr) / 2,
        'Pitch Threshold': pitch_threshold,
        'Pitch F1': pitch_f1,
        'Rhythm Threshold': rhythm_threshold,
        'Rhythm F1': rhythm_f1,
        'Pitch_SE': pitch_se,
        'Rhythm_SE': rhythm_se
    }

def perform_friedman_test(results, feature='Pitch'):
    """
    Perform Friedman test on squared error values across approaches.
    Returns test statistic, p-value, and post-hoc results if significant.
    """
    approaches = list(results.keys())
    se_key = 'Pitch_SE' if feature == 'Pitch' else 'Rhythm_SE'
    
    # Get squared error values for each approach
    se_values = [results[app][se_key] for app in approaches]
    
    # Print squared error statistics for verification
    print(f"\n{feature} Squared Error Statistics:")
    for app, se in zip(approaches, se_values):
        print(f"{app}: Mean = {np.mean(se):.4f}, Std = {np.std(se):.4f}")
    
    # Reshape data for Friedman test
    se_array = np.array(se_values).T  # Transpose to get cases as rows, approaches as columns
    
    # Perform Friedman test
    statistic, p_value = stats.friedmanchisquare(*se_values)
    
    print(f"\nFriedman Test Results for {feature} Squared Errors:")
    print("-" * 50)
    print(f"Statistic: {statistic:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("\nSignificant differences found between approaches!")
        # Perform post-hoc Nemenyi test
        posthoc = posthoc_nemenyi_friedman(se_array)
        print("\nPost-hoc Nemenyi Test p-values:")
        print("-" * 50)
        posthoc.columns = approaches
        posthoc.index = approaches
        print(posthoc.round(4))
        return {
            'statistic': statistic,
            'p_value': p_value,
            'posthoc': posthoc,
            'se_values': dict(zip(approaches, [list(se) for se in se_values])),
            'se_means': dict(zip(approaches, [np.mean(se) for se in se_values])),
            'se_stds': dict(zip(approaches, [np.std(se) for se in se_values]))
        }
    else:
        print("\nNo significant differences found between approaches.")
        return {
            'statistic': statistic,
            'p_value': p_value,
            'posthoc': None,
            'se_values': dict(zip(approaches, [list(se) for se in se_values])),
            'se_means': dict(zip(approaches, [np.mean(se) for se in se_values])),
            'se_stds': dict(zip(approaches, [np.std(se) for se in se_values]))
        }

def plot_mse_comparison(results):
    """Plot MSE comparison across approaches."""
    approaches = list(results.keys())
    pitch_mse = [results[app]['Pitch MSE'] for app in approaches]
    rhythm_mse = [results[app]['Rhythm MSE'] for app in approaches]
    combined_mse = [results[app]['Combined Features MSE'] for app in approaches]
    
    x = np.arange(len(approaches))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    pitch_bars = ax.bar(x - width, pitch_mse, width, label='Pitch MSE', color='skyblue')
    rhythm_bars = ax.bar(x, rhythm_mse, width, label='Rhythm MSE', color='lightgreen')
    combined_bars = ax.bar(x + width, combined_mse, width, label='Combined Features MSE', color='lightcoral')
    
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('MSE Comparison')
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
    autolabel(combined_bars)
    
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
    thresholds = np.arange(1, 100, 1)  # Thresholds as percentages
    
    # Plot for Pitch similarity
    for name, df in similarity_reports.items():
        f1_scores = []
        scores = df['Pitch Similarity'].values  # Keep scores as percentages
        true_labels = df['Binary Ruling'].values
        
        for threshold in thresholds:
            preds = (scores >= threshold).astype(int)
            f1_scores.append(f1_score(true_labels, preds))
        
        best_threshold, best_f1 = find_best_threshold(true_labels, df['Pitch Similarity'])
        ax1.plot(thresholds, f1_scores, label=f'{name}')
        ax1.axvline(x=best_threshold*100, color='gray', linestyle='--', alpha=0.5)
        ax1.axhline(y=best_f1, color='gray', linestyle='--', alpha=0.5)
        ax1.plot(best_threshold*100, best_f1, 'ro', label=f'{name} Best (t={best_threshold*100:.1f}%, F1={best_f1:.2f})')
    
    ax1.set_xlabel('Threshold (%)')
    ax1.set_ylabel('F1-Score')
    ax1.set_title('F1-Score vs Threshold - Pitch Similarity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot for Rhythm similarity
    for name, df in similarity_reports.items():
        f1_scores = []
        scores = df['Rhythm Similarity'].values  # Keep scores as percentages
        true_labels = df['Binary Ruling'].values
        
        for threshold in thresholds:
            preds = (scores >= threshold).astype(int)
            f1_scores.append(f1_score(true_labels, preds))
        
        best_threshold, best_f1 = find_best_threshold(true_labels, df['Rhythm Similarity'])
        ax2.plot(thresholds, f1_scores, label=f'{name}')
        ax2.axvline(x=best_threshold*100, color='gray', linestyle='--', alpha=0.5)
        ax2.axhline(y=best_f1, color='gray', linestyle='--', alpha=0.5)
        ax2.plot(best_threshold*100, best_f1, 'ro', label=f'{name} Best (t={best_threshold*100:.1f}%, F1={best_f1:.2f})')
    
    ax2.set_xlabel('Threshold (%)')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('F1-Score vs Threshold - Rhythm Similarity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def display_pr_analysis_table(similarity_df, feature='Pitch Similarity'):
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
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'Threshold': threshold,
            'True Positives': tp,
            'False Positives': fp,
            'False Negatives': fn,
            'True Negatives': tn,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
    
    print(f"\nPlagiarism Detection Analysis for {feature}")
    print("=" * 100)
    print("If we consider scores above X as plagiarism:")
    print("-" * 100)
    print(f"{'X (threshold)':>12} {'True':>10} {'False':>10} {'False':>10} {'True':>10} {'Accuracy of':>12} {'Coverage of':>12} {'F1':>10}")
    print(f"{'(as %)':>12} {'Pos':>10} {'Pos':>10} {'Neg':>10} {'Neg':>10} {'Detection':>12} {'Plagiarism':>12} {'Score':>10}")
    print("-" * 100)
    
    for row in results:
        threshold_pct = row['Threshold'] * 100
        print(f"{threshold_pct:>12.1f}% {row['True Positives']:>10d} {row['False Positives']:>10d} {row['False Negatives']:>10d} {row['True Negatives']:>10d} {row['Precision']:>12.4f} {row['Recall']:>12.4f} {row['F1-Score']:>10.4f}")
    print("-" * 100)
    print("True Positives (TP)  = Correctly identified plagiarism cases")
    print("False Positives (FP) = Flagged as NOT plagiarism")
    print("False Negatives (FN) = Missed plagiarism cases")
    print("True Negatives (TN)  = Correctly flagged non-plagiarism cases")
    print("Precision = TP / (TP + FP) = Accuracy of plagiarism detection")
    print("Recall = TP / (TP + FN) = Coverage of actual plagiarism cases")

def print_evaluation_results(results, show_pr_analysis=False):
    """Print detailed evaluation results in a formatted table."""
    print("\nEvaluation Results:")
    print("-" * 140)
    headers = ['Approach', 'Pitch MSE', 'Pitch MSE Std', 'Rhythm MSE', 'Rhythm MSE Std', 'Combined Features MSE', 'Combined Features MSE Std', 
              'Pitch AUC', 'Rhythm AUC', 'Avg AUC',
              'Pitch AUC-PR', 'Rhythm AUC-PR', 'Avg AUC-PR',
              'Pitch Thresh', 'Pitch F1', 'Rhythm Thresh', 'Rhythm F1']
    print(f"{headers[0]:<15} " + " ".join(f"{h:>10}" for h in headers[1:]))
    print("-" * 140)
    
    for approach, scores in results.items():
        values = [
            f"{scores['Pitch MSE']:>10.4f}",
            f"{scores['Pitch MSE Std']:>10.4f}",
            f"{scores['Rhythm MSE']:>10.4f}",
            f"{scores['Rhythm MSE Std']:>10.4f}",
            f"{scores['Combined Features MSE']:>10.4f}",
            f"{scores['Combined Features MSE Std']:>10.4f}",
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

def save_squared_error_metrics(results, output_path):
    """Save squared error standard deviation metrics to a CSV file."""
    squared_error_data = []
    for approach, scores in results.items():
        row = {
            'Approach': approach,
            'Pitch_SE_Std': f"{np.std(scores['Pitch_SE']) if 'Pitch_SE' in scores else 0:.4f}",
            'Rhythm_SE_Std': f"{np.std(scores['Rhythm_SE']) if 'Rhythm_SE' in scores else 0:.4f}"
        }
        squared_error_data.append(row)
    
    df = pd.DataFrame(squared_error_data)
    df.to_csv(output_path, index=False)
    print(f"\nSquared error standard deviations saved to: {output_path}")

def save_evaluation_report(results, output_path):
    """Save evaluation results to CSV with proper decimal formatting."""
    # Save squared error metrics to separate file
    se_path = Path(__file__).parent / "squared_error_metrics.csv"
    save_squared_error_metrics(results, se_path)
    
    # Perform Friedman tests
    pitch_friedman = perform_friedman_test(results, 'Pitch')
    rhythm_friedman = perform_friedman_test(results, 'Rhythm')
    
    # Create main results data without MSE metrics
    data = []
    for approach, scores in results.items():
        row = {
            'Approach': approach,
            'Pitch_MSE': f"{scores['Pitch MSE']:.4f}",
            'Rhythm_MSE': f"{scores['Rhythm MSE']:.4f}",
            'Average_MSE': f"{scores['Combined Features MSE']:.4f}",
            'Pitch_AUC': f"{scores['Pitch AUC']:.4f}",
            'Rhythm_AUC': f"{scores['Rhythm AUC']:.4f}",
            'Average_AUC': f"{scores['Average AUC']:.4f}",
            'Pitch_AUC_PR': f"{scores['Pitch AUC-PR']:.4f}",
            'Rhythm_AUC_PR': f"{scores['Rhythm AUC-PR']:.4f}",
            'Average_AUC_PR': f"{scores['Average AUC-PR']:.4f}",
            'Pitch_Threshold': f"{scores['Pitch Threshold']*100:.2f}",
            'Rhythm_Threshold': f"{scores['Rhythm Threshold']*100:.2f}",
            'Pitch_F1': f"{scores['Pitch F1']:.4f}",
            'Rhythm_F1': f"{scores['Rhythm F1']:.4f}"
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save main results
    df.to_csv(output_path, index=False)
    
    # Save Friedman results
    friedman_path = Path(__file__).parent / "friedman_test_results.csv"
    friedman_data = {
        'Feature': ['Pitch', 'Rhythm'],
        'Friedman_Statistic': [
            f"{pitch_friedman['statistic']:.4f}",
            f"{rhythm_friedman['statistic']:.4f}"
        ],
        'Friedman_PValue': [
            f"{pitch_friedman['p_value']:.4f}",
            f"{rhythm_friedman['p_value']:.4f}"
        ],
        'Significant': [
            'Yes' if pitch_friedman['p_value'] < 0.05 else 'No',
            'Yes' if rhythm_friedman['p_value'] < 0.05 else 'No'
        ]
    }
    pd.DataFrame(friedman_data).to_csv(friedman_path, index=False)
    print(f"\nEvaluation results saved to: {output_path}")
    print(f"Friedman test results saved to: {friedman_path}")
    
    # Save post-hoc results if available
    if pitch_friedman['posthoc'] is not None or rhythm_friedman['posthoc'] is not None:
        posthoc_path = Path(__file__).parent / "evaluation_posthoc_results.csv"
        posthoc_data = []
        
        if pitch_friedman['posthoc'] is not None:
            pitch_posthoc = pitch_friedman['posthoc'].reset_index()
            pitch_posthoc['Feature'] = 'Pitch'
            posthoc_data.append(pitch_posthoc)
            
        if rhythm_friedman['posthoc'] is not None:
            rhythm_posthoc = rhythm_friedman['posthoc'].reset_index()
            rhythm_posthoc['Feature'] = 'Rhythm'
            posthoc_data.append(rhythm_posthoc)
        
        if posthoc_data:
            pd.concat(posthoc_data).to_csv(posthoc_path, index=False)
            print(f"Post-hoc analysis results saved to: {posthoc_path}")

def save_individual_squared_errors(similarity_reports, output_path):
    """Save individual squared errors for each case across all approaches to CSV."""
    individual_errors = []
    
    for approach_name, df in similarity_reports.items():
        binary_ruling = df['Binary Ruling'].values
        pitch_scores = df['Pitch Similarity'].values / 100  # Convert percentage to [0,1]
        rhythm_scores = df['Rhythm Similarity'].values / 100
        cases = df['Case'].values if 'Case' in df.columns else [f"Case_{str(i).zfill(3)}" for i in df.index]
        
        # Calculate squared errors for each case and round to 4 decimal places
        pitch_squared_errors = np.round((pitch_scores - binary_ruling) ** 2, decimals=4)
        rhythm_squared_errors = np.round((rhythm_scores - binary_ruling) ** 2, decimals=4)
        
        # Add each case to the list with proper Case_000 format
        for idx in range(len(binary_ruling)):
            individual_errors.append({
                'Approach': approach_name,
                'Case': cases[idx],
                'Ground_Truth': binary_ruling[idx],
                'Pitch_Score': pitch_scores[idx],
                'Rhythm_Score': rhythm_scores[idx],
                'Pitch_Squared_Error': pitch_squared_errors[idx],
                'Rhythm_Squared_Error': rhythm_squared_errors[idx]
            })
    
    # Convert to DataFrame and save to CSV
    error_df = pd.DataFrame(individual_errors)
    error_df.to_csv(output_path, index=False)
    print(f"\nIndividual squared errors saved to: {output_path}")

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
    
    # Save individual squared errors
    errors_path = Path(__file__).parent / "individual_squared_errors.csv"
    save_individual_squared_errors(similarity_reports, errors_path)
    print(f"Results saved to: {report_path}")
    
    while True:
        choice = show_menu()
        
        if choice == '1':
            # Basic Statistics without PR Analysis
            print("\nBasic Evaluation Results:")
            print("-" * 110)
            headers = ['Approach', 'Pitch MSE', 'Pitch MSE Std', 'Rhythm MSE', 'Rhythm MSE Std', 'Combined Features MSE', 'Combined Features MSE Std', 
                      'Pitch AUC-ROC', 'Rhythm AUC-ROC', 'Avg AUC-ROC']
            print(f"{headers[0]:<15} " + " ".join(f"{h:>12}" for h in headers[1:]))
            print("-" * 110)
            
            for approach, scores in results.items():
                values = [
                    f"{scores['Pitch MSE']:>12.4f}",
                    f"{scores['Pitch MSE Std']:>12.4f}",
                    f"{scores['Rhythm MSE']:>12.4f}",
                    f"{scores['Rhythm MSE Std']:>12.4f}",
                    f"{scores['Combined Features MSE']:>12.4f}",
                    f"{scores['Combined Features MSE Std']:>12.4f}",
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
            # Perform Friedman tests
            print("\nPerforming Friedman Tests...")
            pitch_results = perform_friedman_test(results, 'Pitch')
            rhythm_results = perform_friedman_test(results, 'Rhythm')
            input("\nPress Enter to continue...")
        elif choice == '8':
            print_evaluation_results(results, show_pr_analysis=True)
            plot_mse_comparison(results)
            plot_auc_comparison(results)
            plot_roc_curves(similarity_reports)
            plot_pr_curves(similarity_reports)
            plot_f1_threshold_curves(similarity_reports)
            print("\nFriedman Test Analysis:")
            pitch_results = perform_friedman_test(results, 'Pitch')
            rhythm_results = perform_friedman_test(results, 'Rhythm')
        elif choice == '9':
            print("\nExiting evaluation. Goodbye!")
            break
        else:
            print("\nInvalid choice. Please try again.")
        
        if choice != '9':
            input("\nPress Enter to continue...")

def show_menu():
    """Display interactive menu for evaluation options."""
    print("\nEvalution Menu for Similarity Reports of the Approaches")
    print("=" * 40)
    print("1. Show Basic Statistics")
    print("2. Show MSE Comparison")
    print("3. Show AUC-ROC Comparison")
    print("4. Show AUC-PR Comparison")
    print("5. Show Thresholding Analysis")
    print("6. Show Optimal Thresholds for Each Approach via F1-Scoring")
    print("7. Perform Friedman Test Analysis")
    print("8. Generate Full Report (All Metrics)")
    print("9. Exit")
    return input("\nSelect an option (1-9): ")

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

def main():
    """Main execution with interactive menu."""
    interactive_evaluation()

if __name__ == "__main__":
    main()