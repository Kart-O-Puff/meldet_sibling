"""
MelDet Approach: Melodic Similarity Detection using N-gram Analysis and Edit Distance

This module implements a melodic similarity detection approach that combines:
1. N-gram sequence analysis
2. Edit distance computation
3. Logarithmic similarity transformation
4. Diagonal alignment scoring

Key Components:
- Sequence Preprocessing: Converts melodies into relative pitch and rhythm sequences
- Matrix Analysis: Computes edit distances and similarity scores
- Visualization: Provides both detailed and heatmap visualizations
- Scoring: Uses best diagonal alignment for final similarity assessment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Added for heatmap visualization
import ast  # for safely evaluating string representations of lists
from pathlib import Path

def load_sequences_from_library():
    """
    Load and parse the preprocessed melody library from CSV.
    
    The library contains:
    - Relative pitch sequences: Representing melodic intervals
    - Relative rhythm sequences: Representing duration patterns
    - Case information: Grouping related song pairs
    - Rulings: Expert decisions on plagiarism cases
    
    Returns:
    - DataFrame containing parsed sequences and metadata
    """
    try:
        # Get absolute path and verify file exists
        csv_path = Path(__file__).parent / "MCIC_Dataset" / "MCIC_Preprocessed" / "melody_library_symbolic.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Cannot find file: {csv_path}")
        
        # Read CSV with explicit encoding and error handling
        df = pd.read_csv(csv_path, encoding='cp1252', on_bad_lines='skip')
        if df.empty:
            raise ValueError("CSV file is empty")
            
        print(f"Successfully loaded {len(df)} rows from {csv_path}")
        
        # Convert string representations back to actual lists
        df['Relative Pitch'] = df['Relative Pitch'].apply(ast.literal_eval)
        df['Relative Rhythm'] = df['Relative Rhythm'].apply(ast.literal_eval)
        
        # Ensure 'Ruling' column exists
        if 'Category' in df.columns and 'Ruling' not in df.columns:
            df['Ruling'] = df['Category']
            df = df.drop('Category', axis=1)
        
        return df
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure melody_library.csv exists in MCIC_Dataset/MCIC_Preprocessed/")
        raise
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty")
        raise
    except Exception as e:
        print(f"Unexpected error loading CSV: {e}")
        raise

def compare_sequences(seq1, seq2):
    """
    Compare individual n-gram sequences and count differences.
    
    Implementation:
    1. Zips corresponding elements from both sequences
    2. Counts positions where elements differ
    3. Returns total number of differences
    
    This forms the basis for the edit distance calculation.
    """
    return sum(1 for x, y in zip(seq1, seq2) if x != y)

def create_cost_matrix(seq1_grams, seq2_grams):
    """
    Calculate edit distance matrix between two n-gram sequences.
    
    Process:
    1. Initialize matrix of size m x n (m = len(seq1), n = len(seq2))
    2. For each pair of n-grams:
       - Count element-wise differences
       - Store count in corresponding matrix position
    
    Returns:
    - cost_matrix: Edit distances between all n-gram pairs
    - ngram_length: Length of n-grams (for normalization)
    """
    m, n = len(seq1_grams), len(seq2_grams)
    cost_matrix = np.zeros((m, n))
    
    # Get n-gram length from first non-None sequence
    ngram_length = None
    for seq in seq1_grams + seq2_grams:
        if seq is not None:
            ngram_length = len(seq)
            break
            
    if ngram_length is None:
        return cost_matrix
    
    for i in range(m):
        for j in range(n):
            # Count differences between two sequences
            differences = sum(1 for x, y in zip(seq1_grams[i], seq2_grams[j]) 
                            if x != y)
            cost_matrix[i][j] = differences
    
    return cost_matrix, ngram_length

def calculate_log_transform_distance(value, max_d):
    """
    Transform edit distance to similarity score using logarithmic scaling.
    
    Formula: similarity = 1 - log(1 + d) / log(1 + max_d)
    where:
    - d: edit distance value
    - max_d: maximum possible distance (n-gram length)
    
    Properties:
    - Bounded in [0,1] range
    - Non-linear scaling emphasizing smaller differences
    - Preserves relative ordering of distances
    """
    try:
        log_similarity = 1 - (np.log2(1 + value) / np.log2(1 + max_d))
        return round(max(0.0, min(1.0, log_similarity)), 4)
    except:
        return 0.0

def log_transform_distances(distance_matrix, ngram_length):
    """
    Convert entire distance matrix to similarity scores.
    
    Process:
    1. Initialize similarity matrix of same shape
    2. Apply log transform to each distance value
    3. Normalize using n-gram length as max distance
    
    Returns matrix where:
    - 1.0 indicates identical n-grams
    - 0.0 indicates maximum difference
    """
    similarity_matrix = np.zeros_like(distance_matrix, dtype=float)
    
    # Use actual n-gram length as max_d
    max_distance = ngram_length
    
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            distance = distance_matrix[i, j]
            similarity_matrix[i, j] = calculate_log_transform_distance(distance, max_distance)
    
    return similarity_matrix

def visualize_similarity_matrix(matrix, seq1, seq2, output_path):
    """
    Visualize the similarity matrix as a heatmap and highlight the best-matching diagonal.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap='YlOrRd', xticklabels=seq2, yticklabels=seq1)
    
    # Find the diagonal with highest average similarity
    rows, cols = matrix.shape
    max_score = 0
    best_offset = 0
    
    for offset in range(-rows + 1, cols):
        diag = np.diagonal(matrix, offset=offset)
        avg_score = np.mean(diag)
        if avg_score > max_score:
            max_score = avg_score
            best_offset = offset
    
    # Highlight the best diagonal
    if best_offset >= 0:
        plt.plot(range(best_offset, min(cols, rows + best_offset)), 
                range(min(rows, cols - best_offset)), 
                'b--', linewidth=2, label='Best alignment')
    else:
        plt.plot(range(min(cols, rows + best_offset)), 
                range(-best_offset, min(rows, cols - best_offset)), 
                'b--', linewidth=2, label='Best alignment')
    
    plt.legend()
    plt.title('Token Similarity Matrix with Best Alignment')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_matrix_as_table(matrix, seq1, seq2, title, is_similarity=False, similarity_score=None, song1="", song2="", forced_diagonal=None):
    """
    Visualize similarity/distance matrix with detailed values and forced diagonal position.
    """
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

    m, n = matrix.shape
    min_len = min(m, n)
    
    # Use the forced diagonal position if provided
    best_shift = forced_diagonal if forced_diagonal is not None else 0
    
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            cell = table[i+1, j]
            value = matrix[i, j]
            color = cmap(norm(value))
            cell.set_facecolor(color)
            
            # Highlight cells using the forced diagonal
            if m <= n and i < min_len and j == i + best_shift:
                cell.set_edgecolor('red')
                cell.set_linewidth(2)
            elif m > n and j < min_len and i == j + best_shift:
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

def plot_heatmap(matrix, seq1, seq2, title, is_similarity=False, similarity_score=None, song1="", song2="", forced_diagonal=None):
    """Plot similarity/distance matrix as a heatmap with forced diagonal."""
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
    plt.xlabel(f"Song B: {song2}")
    plt.ylabel(f"Song A: {song1}")
    
    plt.xticks(range(len(seq2)), [f'S2_{i}' for i in range(len(seq2))])
    plt.yticks(range(len(seq1)), [f'S1_{i}' for i in range(len(seq1))])
    
    # Use the forced diagonal position
    m, n = matrix.shape
    min_len = min(m, n)
    best_shift = forced_diagonal if forced_diagonal is not None else 0
    
    # Plot the diagonal using the forced position
    if m <= n:
        plt.plot(range(best_shift, best_shift + min_len),
                range(min_len),
                'rx-', markersize=8, linewidth=2, label='Best diagonal')
    else:
        plt.plot(range(min_len),
                range(best_shift, best_shift + min_len),
                'rx-', markersize=8, linewidth=2, label='Best diagonal')
    
    plt.legend()
    
    full_title = f"{title}\n"
    if song1 and song2:
        full_title += f"{song1} vs {song2}\n"
    if similarity_score is not None:
        full_title += f"Similarity Score: {similarity_score * 100:.2f}%"
    plt.title(full_title)
    
    plt.tight_layout()
    plt.show()

def find_best_diagonal(matrix, is_similarity=True):
    """Find the best diagonal alignment in a matrix."""
    m, n = matrix.shape
    min_len = min(m, n)
    max_shift = abs(m - n) + 1
    scores = []
    
    for shift in range(max_shift):
        if m <= n:
            diag = [matrix[i, i + shift] for i in range(min_len)]
        else:
            diag = [matrix[i + shift, i] for i in range(min_len)]
        scores.append(np.mean(diag))
    
    # For similarity matrices, find maximum; for distance matrices, find minimum
    best_shift = scores.index(max(scores) if is_similarity else min(scores))
    return best_shift, max(scores) if is_similarity else min(scores)

def calculate_similarity_score(matrix, forced_shift=None):
    """Calculate similarity score using specified or best diagonal."""
    m, n = matrix.shape
    min_len = min(m, n)
    
    if forced_shift is not None:
        if m <= n:
            diag = [matrix[i, i + forced_shift] for i in range(min_len)]
        else:
            diag = [matrix[i + forced_shift, i] for i in range(min_len)]
        return np.mean(diag)
    
    # If no forced_shift, find best diagonal
    best_shift, best_score = find_best_diagonal(matrix)
    return best_score

def analyze_case(df, case_number, show_visualizations=False):
    """Perform complete similarity analysis for a song pair."""
    case_data = df[df['Case'] == case_number].reset_index(drop=True)
    
    if len(case_data) != 2:
        print(f"Error: Case {case_number} does not have exactly 2 songs")
        return
    
    # Get sequences
    seq1_pitch = case_data.loc[0, 'Relative Pitch']
    seq2_pitch = case_data.loc[1, 'Relative Pitch']
    seq1_rhythm = case_data.loc[0, 'Relative Rhythm']
    seq2_rhythm = case_data.loc[1, 'Relative Rhythm']
    song1_title = case_data.loc[0, 'File Name']
    song2_title = case_data.loc[1, 'File Name']
    
    # Create matrices
    pitch_distances, pitch_ngram_len = create_cost_matrix(seq1_pitch, seq2_pitch)
    rhythm_distances, rhythm_ngram_len = create_cost_matrix(seq1_rhythm, seq2_rhythm)
    
    pitch_similarities = log_transform_distances(pitch_distances, pitch_ngram_len)
    rhythm_similarities = log_transform_distances(rhythm_distances, rhythm_ngram_len)
    
    # Find best diagonal from pitch similarity matrix only
    best_shift = 0
    m, n = pitch_similarities.shape
    min_len = min(m, n)
    max_shift = abs(m - n) + 1
    max_score = 0

    # Calculate best diagonal from pitch matrix
    for shift in range(max_shift):
        if m <= n:
            diag = [pitch_similarities[i, i + shift] for i in range(min_len)]
        else:
            diag = [pitch_similarities[i + shift, i] for i in range(min_len)]
        avg_score = np.mean(diag)
        if avg_score > max_score:
            max_score = avg_score
            best_shift = shift
    
    # Use pitch's best diagonal for both calculations
    pitch_similarity = calculate_similarity_score(pitch_similarities, forced_shift=best_shift)
    rhythm_similarity = calculate_similarity_score(rhythm_similarities, forced_shift=best_shift)
    
    print(f"\nFinal Similarity Scores for {case_number}:")
    print(f"Pitch Similarity: {pitch_similarity * 100:.2f}%")
    print(f"Rhythm Similarity: {rhythm_similarity * 100:.2f}%")
    print(f"Using diagonal position: {best_shift}")
    
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
                                   song1=song1_title, song2=song2_title,
                                   forced_diagonal=best_shift)
                plot_matrix_as_table(similarities, seq1, seq2, 
                                   f"Similarity Matrix Values ({feature})", 
                                   is_similarity=True,
                                   similarity_score=score,
                                   song1=song1_title, song2=song2_title,
                                   forced_diagonal=best_shift)
            
            if viz_type in ['2', '3']:
                plot_heatmap(distances, seq1, seq2,
                           f"Edit Distance Heatmap ({feature})",
                           song1=song1_title, song2=song2_title,
                           forced_diagonal=best_shift)
                plot_heatmap(similarities, seq1, seq2,
                           f"Similarity Heatmap ({feature})",
                           is_similarity=True,
                           similarity_score=score,
                           song1=song1_title, song2=song2_title,
                           forced_diagonal=best_shift)

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
        
        # Create cost matrices and transform to similarities
        pitch_matrix, pitch_ngram_len = create_cost_matrix(seq1_pitch, seq2_pitch)
        rhythm_matrix, rhythm_ngram_len = create_cost_matrix(seq1_rhythm, seq2_rhythm)
        
        pitch_similarities = log_transform_distances(pitch_matrix, pitch_ngram_len)
        rhythm_similarities = log_transform_distances(rhythm_matrix, rhythm_ngram_len)
        
        # Find best diagonal from pitch similarity matrix
        best_shift, _ = find_best_diagonal(pitch_similarities, is_similarity=True)
        
        # Calculate scores using the same diagonal for both pitch and rhythm
        pitch_similarity = calculate_similarity_score(pitch_similarities, forced_shift=best_shift)
        rhythm_similarity = calculate_similarity_score(rhythm_similarities, forced_shift=best_shift)
        
        results.append({
            'Case': case,
            'Ruling': ruling,
            'Song A': song1_title,
            'Song B': song2_title,
            'Pitch Similarity': pitch_similarity,
            'Rhythm Similarity': rhythm_similarity
        })
        
        if show_plots:
            plot_matrix_as_table(pitch_matrix, seq1_pitch, seq2_pitch, 
                           f"Pitch Cost Matrix - {case} ({ruling})\nSimilarity Score: {pitch_similarity}",
                           is_similarity=True,
                           song1=song1_title, song2=song2_title,
                           forced_diagonal=best_shift)
            plot_matrix_as_table(rhythm_matrix, seq1_rhythm, seq2_rhythm, 
                           f"Rhythm Cost Matrix - {case} ({ruling})\nSimilarity Score: {rhythm_similarity}",
                           is_similarity=True,
                           song1=song1_title, song2=song2_title,
                           forced_diagonal=best_shift)
    
    return results

def interactive_menu(df):
    """
    Interactive menu for user interaction.
    
    Options:
    1. Analyze individual cases with or without visualizations
    2. List available cases
    3. Exit the program
    """
    while True:
        print("\nMelDet Analysis Options:")
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

def save_similarity_report(results, output_path):
    """
    Generate comprehensive analysis report.
    
    Report Contents:
    1. Case identifiers
    2. Expert rulings (original and binary)
    3. Song pair information
    4. Similarity scores (as percentages)
    
    Format:
    - CSV file with sorted cases
    - Standardized column ordering
    """
    df = pd.DataFrame(results)
    
    # Convert similarity scores to percentages
    df['Pitch Similarity'] = df['Pitch Similarity'] * 100
    df['Rhythm Similarity'] = df['Rhythm Similarity'] * 100
    
    # Add binary ruling column
    df['Binary Ruling'] = (df['Ruling'] == 'Plagiarism').astype(int)
    # Extract case numbers and sort using raw string for regex
    df['Case_Num'] = df['Case'].str.extract(r'(\d+)').astype(int)
    df = df.sort_values('Case_Num')
    df = df.drop('Case_Num', axis=1)
    # Reorder columns to put Binary Ruling after Ruling
    columns = ['Case', 'Ruling', 'Binary Ruling', 'Song A', 'Song B', 'Pitch Similarity', 'Rhythm Similarity']
    df = df[columns]
    df.to_csv(output_path, index=False)
    print(f"Similarity report saved to: {output_path}")

def main():
    """
    Main execution flow of the MelDet analysis system.
    
    Workflow:
    1. Load preprocessed melody library
    2. Analyze all cases
    3. Generate similarity report
    4. Start interactive analysis session
    
    The interactive menu allows for:
    - Individual case analysis
    - Visualization options
    - Case listing
    """
    # Load sequences from library
    df = load_sequences_from_library()
    
    # Analyze all cases without showing plots
    print("Analyzing all cases...")
    results = analyze_all_cases(df, show_plots=False)
    
    # Save results to CSV in MelDet root directory
    report_path = Path(__file__).parent / "similarity_report_meldet.csv"
    save_similarity_report(results, report_path)
    
    print("\nProcessing completed.")
    
    # Start interactive menu
    interactive_menu(df)

if __name__ == "__main__":
    main()
