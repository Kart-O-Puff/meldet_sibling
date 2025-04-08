vision parameter
    metrics = {
        'Accuracy': accuracy_score(ground_truth, binary_predictions),
        'Precision': precision_score(ground_truth, binary_predictions, average='binary', zero_division=0),
        'Recall': recall_score(ground_truth, binary_predictions, average='binary', zero_division=0),
        'F1': f1_score(ground_truth, binary_predictions, average='binary', zero_division=0)
    }
    
    return metrics

def plot_roc_curve(similarity_sco