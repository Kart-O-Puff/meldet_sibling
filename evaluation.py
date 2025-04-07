import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

def load_similarity_reports():
    """Load similarity reports from all three approaches."""
    base_path = Path(__file__).parent
    
    reports = {
        'MelDet': pd.read_csv(base_path / 'similarity_report_meldet.csv'),
        'ModTverskyEqual': pd.read_csv(base_path / 'similarity_report_modtverskyequal.csv'),
        'SumCommon': pd.read_csv(base_path / 'similarity_report_sumcommon.csv')
    }
    
    return reports

def calculate_metrics(predictions, ground_truth, threshold=0.5):
    """
    Calculate evaluation metrics for binary classification.
    
    Parameters Example:
        predictions = [0.8, 0.2, 0.6, 0.3]  # Similarity scores from approaches
        ground_truth = [1, 0, 1, 0]         # Binary rulings from reports
        threshold = 0.5                      # Decision boundary
    
    Binary Conversion Logic:
        Similarity Score -> Binary Prediction
        0.8 >= 0.5 -> 1    (Correctly predicts Plagiarism)
        0.2 < 0.5  -> 0    (Correctly predicts Non-Plagiarism)
        0.6 >= 0.5 -> 1    (Correctly predicts Plagiarism)
        0.3 < 0.5  -> 0    (Correctly predicts Non-Plagiarism)
    
    Metric Calculations:
    1. Accuracy: (TP + TN) / Total
       Example: (2 + 2) / 4 = 1.0 (100% accurate)
       - TP (True Positive): Predicted 1 when actually 1
       - TN (True Negative): Predicted 0 when actually 0
    
    2. Precision: TP / (TP + FP)
       Example: 2 / 2 = 1.0 (No false plagiarism claims)
       - High precision means reliable plagiarism detection
    
    3. Recall: TP / (TP + FN)
       Example: 2 / 2 = 1.0 (Caught all plagiarism cases)
       - High recall means catching most/all plagiarism
    
    4. F1: 2 * (Precision * Recall) / (Precision + Recall)
       Example: 2 * (1.0 * 1.0) / (1.0 + 1.0) = 1.0
       - Balances precision and recall
    
    Real-world Example:
    Similarity Scores: [0.95, 0.15, 0.85, 0.25, 0.55]
    Ground Truth:      [1,    0,    1,    0,    1   ]
    Binary Pred:       [1,    0,    1,    0,    1   ]
    
    - Perfect prediction: All similarity scores align with ground truth
    - High scores (>0.5) correctly predict plagiarism cases
    - Low scores (<0.5) correctly predict non-plagiarism cases
    """
    # Convert predictions to binary based on threshold
    binary_predictions = (predictions >= threshold).astype(int)
    
    # Calculate metrics for both classes
    metrics = {
        'Accuracy': accuracy_score(ground_truth, binary_predictions),
        'Precision': precision_score(ground_truth, binary_predictions, average='binary'),
        'Recall': recall_score(ground_truth, binary_predictions, average='binary'),
        'F1': f1_score(ground_truth, binary_predictions, average='binary')
    }
    
    return metrics

def plot_roc_curve(similarity_scores, binary_ruling, title):
    """Plot ROC curve and calculate AUC."""
    fpr, tpr, _ = roc_curve(binary_ruling, similarity_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {title}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return roc_auc

def evaluate_approach(similarity_df, approach_name):
    """Evaluate pitch and rhythm similarity scores against binary rulings from report."""
    # Get binary rulings from similarity report
    binary_ruling = similarity_df['Binary Ruling'].values
    
    # Calculate and plot ROC curves
    print(f"\nGenerating ROC curves for {approach_name}...")
    pitch_auc = plot_roc_curve(
        similarity_df['Pitch Similarity'],
        binary_ruling,
        f'{approach_name} - Pitch Feature'
    )
    rhythm_auc = plot_roc_curve(
        similarity_df['Rhythm Similarity'],
        binary_ruling,
        f'{approach_name} - Rhythm Feature'
    )
    
    # Evaluate metrics
    pitch_results = {
        'Approach': approach_name,
        'Feature': 'Pitch',
        'AUC': round(pitch_auc, 4),
        **calculate_metrics(
            similarity_df['Pitch Similarity'], 
            binary_ruling
        )
    }
    
    rhythm_results = {
        'Approach': approach_name,
        'Feature': 'Rhythm',
        'AUC': round(rhythm_auc, 4),
        **calculate_metrics(
            similarity_df['Rhythm Similarity'], 
            binary_ruling
        )
    }
    
    return pitch_results, rhythm_results

def save_evaluation_report(results, output_path):
    """Save all evaluation results to a single CSV file."""
    # Prepare rows for DataFrame
    all_results = []
    for approach_results in results:
        all_results.append(approach_results[0])  # Pitch metrics
        all_results.append(approach_results[1])  # Rhythm metrics
    
    # Create DataFrame and save
    df = pd.DataFrame(all_results)
    df.to_csv(output_path, index=False)
    print(f"\nEvaluation results saved to: {output_path}")

def print_separate_evaluation_results(results):
    """Print evaluation results separately for pitch and rhythm."""
    print("\n=== Pitch Feature Evaluation ===")
    pitch_df = pd.DataFrame([r[0] for r in results])
    print(pitch_df.to_string(index=False))
    
    print("\n=== Rhythm Feature Evaluation ===")
    rhythm_df = pd.DataFrame([r[1] for r in results])
    print(rhythm_df.to_string(index=False))

def main():
    """Main execution comparing similarity scores against binary rulings from reports."""
    # Load similarity reports
    similarity_reports = load_similarity_reports()
    
    # Evaluate each approach
    results = []
    for approach_name, report_df in similarity_reports.items():
        pitch_result, rhythm_result = evaluate_approach(report_df, approach_name)
        results.append((pitch_result, rhythm_result))
    
    # Print separated results
    print_separate_evaluation_results(results)
    
    # Save combined report
    report_path = Path(__file__).parent / "evaluation_report.csv"
    save_evaluation_report(results, report_path)

if __name__ == "__main__":
    main()
