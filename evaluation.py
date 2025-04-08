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
    Calculate evaluation metrics between similarity scores and binary ruling.
    Handles zero division cases for precision, recall, and F1 score.
    """
    # Convert predictions to binary based on threshold
    binary_predictions = (predictions >= threshold).astype(int)
    
    # Calculate metrics with zero_division parameter
    metrics = {
        'Accuracy': accuracy_score(ground_truth, binary_predictions),
        'Precision': precision_score(ground_truth, binary_predictions, average='binary', zero_division=0),
        'Recall': recall_score(ground_truth, binary_predictions, average='binary', zero_division=0),
        'F1': f1_score(ground_truth, binary_predictions, average='binary', zero_division=0)
    }
    
    return metrics

def plot_roc_curve(similarity_scores, binary_ruling, title, song1="", song2="", score=None):
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
    
    # Enhanced title with song information and percentage score
    full_title = f'ROC Curve - {title}\n'
    if song1 and song2:
        full_title += f'{song1} vs {song2}\n'
    if score is not None:
        full_title += f'Similarity Score: {score:.2f}%'
    plt.title(full_title)
    
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return roc_auc

def evaluation_visualization_menu():
    """Display menu for ROC curve visualization options."""
    print("\nROC Curve Visualization Options:")
    print("1. Show Pitch Analysis")
    print("2. Show Rhythm Analysis")
    print("3. Show Both Analyses")
    
    while True:
        choice = input("\nEnter your choice (1-3): ")
        if choice in ['1', '2', '3']:
            return choice
        print("Invalid choice! Please try again.")

def evaluate_approach(similarity_df, approach_name, show_plots=False):
    """Evaluate pitch and rhythm similarity scores against binary rulings from report."""
    # Get binary rulings and song titles from similarity report
    binary_ruling = similarity_df['Binary Ruling'].values
    song1_titles = similarity_df['Song 1'].values
    song2_titles = similarity_df['Song 2'].values
    
    if show_plots:
        print(f"\nGenerating ROC curves for {approach_name}...")
        viz_choice = evaluation_visualization_menu()
        
        for idx, (song1, song2) in enumerate(zip(song1_titles, song2_titles)):
            # Plot Pitch ROC curve if selected
            if viz_choice in ['1', '3']:
                pitch_auc = plot_roc_curve(
                    similarity_df['Pitch Similarity'],
                    binary_ruling,
                    f'{approach_name} - Pitch Feature',
                    song1, song2,
                    similarity_df['Pitch Similarity'].iloc[idx] * 100
                )
            
            # Plot Rhythm ROC curve if selected
            if viz_choice in ['2', '3']:
                rhythm_auc = plot_roc_curve(
                    similarity_df['Rhythm Similarity'],
                    binary_ruling,
                    f'{approach_name} - Rhythm Feature',
                    song1, song2,
                    similarity_df['Rhythm Similarity'].iloc[idx] * 100
                )
    else:
        # Calculate AUC without plotting
        _, tpr_pitch, _ = roc_curve(binary_ruling, similarity_df['Pitch Similarity'])
        pitch_auc = auc(_, tpr_pitch)
        _, tpr_rhythm, _ = roc_curve(binary_ruling, similarity_df['Rhythm Similarity'])
        rhythm_auc = auc(_, tpr_rhythm)
    
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

def evaluation_menu():
    """Display menu for evaluation options."""
    print("\nEvaluation Options:")
    print("1. Show evaluation with ROC curve visualizations")
    print("2. Show evaluation metrics only")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ")
        if choice in ['1', '2', '3']:
            return choice
        print("Invalid choice! Please try again.")

def main():
    """Main execution comparing similarity scores against binary rulings from reports."""
    # Load similarity reports
    similarity_reports = load_similarity_reports()
    
    while True:
        choice = evaluation_menu()
        
        if choice == '3':
            print("Exiting...")
            break
        
        show_plots = (choice == '1')
        
        # Evaluate each approach
        results = []
        for approach_name, report_df in similarity_reports.items():
            pitch_result, rhythm_result = evaluate_approach(
                report_df, approach_name, show_plots)
            results.append((pitch_result, rhythm_result))
        
        # Print separated results
        print_separate_evaluation_results(results)
        
        # Save combined report
        report_path = Path(__file__).parent / "evaluation_report.csv"
        save_evaluation_report(results, report_path)
        
        if choice == '2':
            break  # Exit after showing metrics only

if __name__ == "__main__":
    main()
