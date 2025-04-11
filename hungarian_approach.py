"""
Hungarian Algorithm Approach: Melodic Similarity Detection using N-gram Analysis and Hungarian Algorithm

This module implements a melodic similarity detection approach that combines:
1. N-gram sequence analysis
2. Edit distance computation
3. Logarithmic similarity transformation
4. Hungarian Algorithm for optimal matching

Key Components:
- Sequence Preprocessing: Converts melodies into relative pitch and rhythm sequences
- Matrix Analysis: Computes edit distances and similarity scores
- Visualization: Provides both detailed and heatmap visualizations
- Scoring: Uses Hungarian Algorithm for optimal assignment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from pathlib import Path
from scipy.optimize import linear_sum_assignment

def load_sequences_from_library():
    """Load and parse the preprocessed melody library from CSV."""
    try:
        csv_path = Path(__file__).parent / "MCIC_Dataset" / "MCIC_Preprocessed" / "melody_library_symbolic.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Cannot find file: {csv_path}")
        
        df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
        if df.empty:
            raise ValueError("CSV file is empty")
            
        print(f"Successfully loaded {len(df)} rows from {csv_path}")
        
        df['Relative Pitch'] = df['Relative Pitch'].apply(ast.literal_eval)
        df['Relative Rhythm'] = df['Relative Rhythm'].apply(ast.literal_eval)
        
        if 'Category' in df.columns and 'Ruling' not in df.columns:
            df['Ruling'] = df['Category']
            df = df.drop('Category', axis=1)
        
        return df
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise

def compare_sequences(seq1, seq2):
    """Compare individual n-gram sequences and count differences."""
    return sum(1 for x, y in zip(seq1, seq2) if x != y)

def create_cost_matrix(seq1_grams, seq2_grams):
    """Calculate edit distance matrix between two n-gram sequences."""
    m, n = len(seq1_grams), len(seq2_grams)
    cost_matrix = np.zeros((m, n))
    
    ngram_length = None
    for seq in seq1_grams + seq2_grams:
        if seq is not None:
            ngram_length = len(seq)
            break
            
    if ngram_length is None:
        return cost_matrix, 0
    
    for i in range(m):
        for j in range(n):
            differences = sum(1 for x, y in zip(seq1_grams[i], seq2_grams[j]) 
                            if x != y)
            cost_matrix[i][j] = differences
    
    return cost_matrix, ngram_length

def calculate_log_transform_distance(value, max_d):
    """Transform edit distance to similarity score using logarithmic scaling."""
    try:
        log_similarity = 1 - (np.log2(1 + value) / np.log2(1 + max_d))
        return round(max(0.0, min(1.0, log_similarity)), 4)
    except:
        return 0.0

def log_transform_distances(distance_matrix, ngram_length):
    """Convert entire distance matrix to similarity scores."""
    similarity_matrix = np.zeros_like(distance_matrix, dtype=float)
    max_distance = ngram_length
    
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            distance = distance_matrix[i, j]
            similarity_matrix[i, j] = calculate_log_transform_distance(distance, max_distance)
    
    return similarity_matrix

def calculate_hungarian_similarity(matrix):
    """
    Calculate similarity score using the Hungarian Algorithm.
    
    The Hungarian Algorithm finds the optimal assignment that maximizes
    the total similarity score between two sequences.
    """
    # For similarity matrix, we want to maximize, so we negate the values
    # since linear_sum_assignment minimizes by default
    row_ind, col_ind = linear_sum_assignment(-matrix)
    
    # Calculate average similarity from optimal assignment
    optimal_similarities = matrix[row_ind, col_ind]
    return np.mean(optimal_similarities)

def plot_matrix_as_table(matrix, seq1, seq2, title, is_similarity=False, similarity_score=None, song1="", song2=""):
    """Plot matrix values as a visual table using matplotlib."""
    fig, (ax_table, ax_colorbar) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 0.2]}, figsize=(14, 8))
    ax_table.axis('tight')
    ax_table.axis('off')
    
    col_labels = [f'S2_{i}:{val}' for i, val in enumerate(seq2)]
    row_labels = [f'S1_{i}:{val}' for i, val in enumerate(seq1)]
    cell_text = [[f'{val:.4f}' for val in row] for row in matrix]
    
    table = ax_table.table(cellText=cell_text,
                          rowLabels=row_labels,
                          colLabels=col_labels,
                          cellLoc='center',
                          loc='center')
    
    if is_similarity:
        norm = plt.Normalize(vmin=0, vmax=1)
        colorbar_label = 'Similarity Value\n(0: Most Different, 1: Most Similar)'
        cmap = plt.cm.viridis
    else:
        norm = plt.Normalize(vmin=0, vmax=7)
        colorbar_label = 'Edit Distance\n(0: Most Similar, 7: Most Different)'
        cmap = plt.cm.viridis_r
    
    # Find optimal assignment using Hungarian Algorithm
    row_ind, col_ind = linear_sum_assignment(-matrix if is_similarity else matrix)
    
    # Color cells and highlight optimal assignment
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            cell = table[i+1, j]
            value = matrix[i, j]
            color = cmap(norm(value))
            cell.set_facecolor(color)
            
            # Highlight cells that are part of the optimal assignment
            if i in row_ind and j == col_ind[list(row_ind).index(i)]:
                cell.set_edgecolor('red')
                cell.set_linewidth(2)
    
    cmap_obj = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(cmap_obj, cax=ax_colorbar, label=colorbar_label)
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    full_title = f"{title}\n"
    if song1 and song2:
        full_title += f"{song1} vs {song2}\n"
    if similarity_score is not None:
        full_title += f"Similarity Score: {similarity_score * 100:.2f}%"
    ax_table.set_title(full_title)
    
    plt.tight_layout()
    plt.show()

def plot_heatmap(matrix, seq1, seq2, title, is_similarity=False, similarity_score=None, song1="", song2=""):
    """Plot similarity/distance matrix as a heatmap without detailed values."""
    plt.figure(figsize=(10, 8))
    
    if is_similarity:
        vmin, vmax = 0, 1
        cmap = 'viridis'
        cbar_label = 'Similarity Value\n(0: Most Different, 1: Most Similar)'
    else:
        vmin, vmax = 0, 7
        cmap = 'viridis_r'
        cbar_label = 'Edit Distance\n(0: Most Similar, 7: Most Different)'
    
    im = plt.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=cbar_label)
    
    plt.xticks(range(len(seq2)), [f'S2_{i}' for i in range(len(seq2))])
    plt.yticks(range(len(seq1)), [f'S1_{i}' for i in range(len(seq1))])
    
    # Find and plot optimal assignment using Hungarian Algorithm
    row_ind, col_ind = linear_sum_assignment(-matrix if is_similarity else matrix)
    plt.plot(col_ind, row_ind, 'rx-', markersize=8, linewidth=2, label='Optimal assignment')
    
    plt.legend()
    
    full_title = f"{title}\n"
    if song1 and song2:
        full_title += f"{song1} vs {song2}\n"
    if similarity_score is not None:
        full_title += f"Similarity Score: {similarity_score * 100:.2f}%"
    plt.title(full_title)
    
    plt.tight_layout()
    plt.show()

def analyze_case(df, case_number, show_visualizations=False):
    """Perform complete similarity analysis for a song pair."""
    case_data = df[df['Case'] == case_number].reset_index(drop=True)
    
    if len(case_data) != 2:
        print(f"Error: Case {case_number} does not have exactly 2 songs")
        return
    
    seq1_pitch = case_data.loc[0, 'Relative Pitch']
    seq2_pitch = case_data.loc[1, 'Relative Pitch']
    seq1_rhythm = case_data.loc[0, 'Relative Rhythm']
    seq2_rhythm = case_data.loc[1, 'Relative Rhythm']
    song1_title = case_data.loc[0, 'File Name']
    song2_title = case_data.loc[1, 'File Name']
    
    pitch_distances, pitch_ngram_len = create_cost_matrix(seq1_pitch, seq2_pitch)
    rhythm_distances, rhythm_ngram_len = create_cost_matrix(seq1_rhythm, seq2_rhythm)
    
    pitch_similarities = log_transform_distances(pitch_distances, pitch_ngram_len)
    rhythm_similarities = log_transform_distances(rhythm_distances, rhythm_ngram_len)
    
    pitch_similarity = calculate_hungarian_similarity(pitch_similarities)
    rhythm_similarity = calculate_hungarian_similarity(rhythm_similarities)
    
    print(f"\nFinal Similarity Scores for {case_number}:")
    print(f"Pitch Similarity: {pitch_similarity * 100:.2f}%")
    print(f"Rhythm Similarity: {rhythm_similarity * 100:.2f}%")
    
    if show_visualizations:
        print("\nVisualization Options:")
        print("1. Show detailed value tables")
        print("2. Show heatmaps only")
        print("3. Show both")
        viz_type = input("Enter visualization type (1-3): ")
        
        for feature, distances, similarities, seq1, seq2, score in [
            ("Pitch", pitch_distances, pitch_similarities, seq1_pitch, seq2_pitch, pitch_similarity),
            ("Rhythm", rhythm_distances, rhythm_similarities, seq1_rhythm, seq2_rhythm, rhythm_similarity)
        ]:
            print(f"\n{feature} Analysis:")
            
            if viz_type in ['1', '3']:
                plot_matrix_as_table(distances, seq1, seq2, 
                                   f"Edit Distance Matrix Values ({feature})",
                                   song1=song1_title, song2=song2_title)
                plot_matrix_as_table(similarities, seq1, seq2, 
                                   f"Similarity Matrix Values ({feature})", 
                                   is_similarity=True,
                                   similarity_score=score,
                                   song1=song1_title, song2=song2_title)
            
            if viz_type in ['2', '3']:
                plot_heatmap(distances, seq1, seq2,
                           f"Edit Distance Heatmap ({feature})",
                           song1=song1_title, song2=song2_title)
                plot_heatmap(similarities, seq1, seq2,
                           f"Similarity Heatmap ({feature})",
                           is_similarity=True,
                           similarity_score=score,
                           song1=song1_title, song2=song2_title)

def analyze_all_cases(df, show_plots=False):
    """Analyze all cases in the dataset."""
    cases = sorted(df['Case'].unique())
    results = []
    
    for case in cases:
        case_data = df[df['Case'] == case].reset_index(drop=True)
        
        if len(case_data) != 2:
            print(f"Error: Case {case} does not have exactly 2 songs")
            continue
        
        ruling = case_data.loc[0, 'Ruling']
        song1_title = case_data.loc[0, 'File Name']
        song2_title = case_data.loc[1, 'File Name']
        seq1_pitch = case_data.loc[0, 'Relative Pitch']
        seq2_pitch = case_data.loc[1, 'Relative Pitch']
        seq1_rhythm = case_data.loc[0, 'Relative Rhythm']
        seq2_rhythm = case_data.loc[1, 'Relative Rhythm']
        
        pitch_matrix, pitch_ngram_len = create_cost_matrix(seq1_pitch, seq2_pitch)
        rhythm_matrix, rhythm_ngram_len = create_cost_matrix(seq1_rhythm, seq2_rhythm)
        
        pitch_similarities = log_transform_distances(pitch_matrix, pitch_ngram_len)
        rhythm_similarities = log_transform_distances(rhythm_matrix, rhythm_ngram_len)
        
        pitch_similarity = calculate_hungarian_similarity(pitch_similarities)
        rhythm_similarity = calculate_hungarian_similarity(rhythm_similarities)
        
        results.append({
            'Case': case,
            'Ruling': ruling,
            'Song 1': song1_title,
            'Song 2': song2_title,
            'Pitch Similarity': pitch_similarity,
            'Rhythm Similarity': rhythm_similarity
        })
        
        if show_plots:
            plot_matrix_as_table(pitch_matrix, seq1_pitch, seq2_pitch, 
                           f"Pitch Cost Matrix - {case} ({ruling})\nSimilarity Score: {pitch_similarity}",
                           is_similarity=True,
                           song1=song1_title, song2=song2_title)
            plot_matrix_as_table(rhythm_matrix, seq1_rhythm, seq2_rhythm, 
                           f"Rhythm Cost Matrix - {case} ({ruling})\nSimilarity Score: {rhythm_similarity}",
                           is_similarity=True,
                           song1=song1_title, song2=song2_title)
    
    return results

def save_similarity_report(results, output_path):
    """Generate comprehensive analysis report."""
    df = pd.DataFrame(results)
    df['Pitch Similarity'] = df['Pitch Similarity'] * 100
    df['Rhythm Similarity'] = df['Rhythm Similarity'] * 100
    df['Binary Ruling'] = (df['Ruling'] == 'Plagiarism').astype(int)
    
    df['Case_Num'] = df['Case'].str.extract(r'(\d+)').astype(int)
    df = df.sort_values('Case_Num')
    df = df.drop('Case_Num', axis=1)
    
    columns = ['Case', 'Ruling', 'Binary Ruling', 'Song 1', 'Song 2', 'Pitch Similarity', 'Rhythm Similarity']
    df = df[columns]
    df.to_csv(output_path, index=False)
    print(f"Similarity report saved to: {output_path}")

def interactive_menu(df):
    """Interactive menu for user interaction."""
    while True:
        print("\nHungarian Algorithm Analysis Options:")
        print("1. Show case analysis (with visualizations)")
        print("2. Show case analysis (without visualizations)")
        print("3. List available cases")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            case = input("Enter case number (e.g., Case_006): ")
            if case in df['Case'].unique():
                analyze_case(df, case, show_visualizations=True)
            else:
                print("Invalid case number!")
        elif choice == '2':
            case = input("Enter case number (e.g., Case_006): ")
            if case in df['Case'].unique():
                analyze_case(df, case, show_visualizations=False)
            else:
                print("Invalid case number!")
        elif choice == '3':
            print("\nAvailable cases:")
            for case in sorted(df['Case'].unique()):
                print(f"- {case}")
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice!")

def main():
    """Main execution flow of the Hungarian Algorithm analysis system."""
    df = load_sequences_from_library()
    
    print("Analyzing all cases...")
    results = analyze_all_cases(df, show_plots=False)
    
    report_path = Path(__file__).parent / "similarity_report_hungarian.csv"
    save_similarity_report(results, report_path)
    
    print("\nProcessing completed.")
    interactive_menu(df)

if __name__ == "__main__":
    main()