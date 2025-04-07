import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast  # for safely evaluating string representations of lists
from pathlib import Path

def load_sequences_from_library():
    """Load and parse the melody library CSV file."""
    try:
        # Get absolute path and verify file exists
        csv_path = Path(__file__).parent / "MCIC_Dataset" / "MCIC_Preprocessed" / "melody_library.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Cannot find file: {csv_path}")
        
        # Read CSV with explicit encoding and error handling
        df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
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
    Compare two individual sequences (n-grams) and count differences.
    Returns the number of positions where elements differ.
    """
    return sum(1 for x, y in zip(seq1, seq2) if x != y)

def create_cost_matrix(seq1_grams, seq2_grams):
    """Calculate edit distance matrix between two sets of n-gram pairs."""
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
    Calculate normalized similarity using logarithmic transform.
    
    Parameters:
        value: The raw distance value
        max_d: Maximum possible distance (length of n-gram sequence)
    """
    try:
        log_similarity = 1 - (np.log2(1 + value) / np.log2(1 + max_d))
        return round(max(0.0, min(1.0, log_similarity)), 4)
    except:
        return 0.0

def log_transform_distances(distance_matrix, ngram_length):
    """Transform edit distances to similarity scores using logarithmic transform."""
    similarity_matrix = np.zeros_like(distance_matrix, dtype=float)
    
    # Use actual n-gram length as max_d
    max_distance = ngram_length
    
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            distance = distance_matrix[i, j]
            similarity_matrix[i, j] = calculate_log_transform_distance(distance, max_distance)
    
    return similarity_matrix

def plot_matrix_as_table(matrix, seq1, seq2, title, is_similarity=False, similarity_score=None):
    """Plot matrix values as a visual table using matplotlib."""
    fig, (ax_table, ax_colorbar) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 0.2]}, figsize=(14, 8))
    ax_table.axis('tight')
    ax_table.axis('off')
    
    # Create row and column labels
    col_labels = [f'S2_{i}:{val}' for i, val in enumerate(seq2)]
    row_labels = [f'S1_{i}:{val}' for i, val in enumerate(seq1)]
    
    # Format matrix values
    cell_text = [[f'{val:.4f}' for val in row] for row in matrix]
    
    # Create table
    table = ax_table.table(cellText=cell_text,
                          rowLabels=row_labels,
                          colLabels=col_labels,
                          cellLoc='center',
                          loc='center')
    
    # Set up color mapping
    if is_similarity:
        norm = plt.Normalize(vmin=0, vmax=1)
        colorbar_label = 'Similarity Value\n(0: Most Different, 1: Most Similar)'
    else:
        norm = plt.Normalize(vmin=0, vmax=7)
        colorbar_label = 'Edit Distance\n(0: No Differences, 7: All Different)'
    
    # Color cells and highlight diagonal
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            cell = table[i+1, j]
            value = matrix[i, j]
            color = plt.cm.viridis(norm(value))
            cell.set_facecolor(color)
            
            if i == j:
                cell.set_edgecolor('red')
                cell.set_linewidth(2)
    
    # Add colorbar
    cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
    fig.colorbar(cmap, cax=ax_colorbar, label=colorbar_label)
    
    # Adjust table properties
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    if similarity_score is not None:
        title = f"{title}\nSimilarity Score: {similarity_score:.4f}"
    ax_table.set_title(title)
    
    plt.tight_layout()
    plt.show()

def plot_heatmap(matrix, seq1, seq2, title, is_similarity=False, similarity_score=None):
    """Plot similarity/distance matrix as a heatmap without detailed values."""
    plt.figure(figsize=(10, 8))
    
    # Set up color mapping
    if is_similarity:
        vmin, vmax = 0, 1
        cmap = 'viridis'
        cbar_label = 'Similarity Value\n(0: Most Different, 1: Most Similar)'
    else:
        vmin, vmax = 0, 7
        cmap = 'viridis'
        cbar_label = 'Edit Distance\n(0: No Differences, 7: All Different)'
    
    # Create heatmap
    im = plt.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add colorbar
    plt.colorbar(im, label=cbar_label)
    
    # Add labels
    plt.xticks(range(len(seq2)), [f'S2_{i}' for i in range(len(seq2))])
    plt.yticks(range(len(seq1)), [f'S1_{i}' for i in range(len(seq1))])
    
    # Highlight diagonal
    for i in range(min(len(seq1), len(seq2))):
        plt.plot(i, i, 'rx', markersize=8)
    
    if similarity_score is not None:
        title = f"{title}\nSimilarity Score: {similarity_score:.4f}"
    plt.title(title)
    
    plt.tight_layout()
    plt.show()

def calculate_similarity_score(matrix):
    """
    Calculate final similarity score from normalized cost matrix.
    
    Why Use Diagonal Elements?
    - Diagonal represents alignment of corresponding elements
    - In similarity matrix, diagonal shows how well sequences align
      at each position when directly compared
    - High diagonal values indicate good alignment at those positions
    
    Score Calculation:
    - Takes mean of diagonal elements because:
      * Diagonal contains position-wise similarity scores
      * Mean gives overall similarity accounting for all positions
      * Results in single [0,1] score where:
        - 1.0 means perfect alignment
        - 0.0 means no alignment
    """
    diagonal = np.diagonal(matrix)
    return np.mean(diagonal)

def analyze_case(df, case_number, show_visualizations=False):
    """
    Complete similarity analysis process for a case:
    
    1. Extract Sequences:
       - Gets relative pitch and rhythm sequences for both songs
    
    2. Compute Edit Distance:
       - Calculates Levenshtein distance matrices
    
    3. Create Similarity Matrices:
       - Normalizes distances with Log Transformation to form similarity scores
       - Higher values indicate more similarity
    
    4. Calculate Final Scores:
       - Takes mean of diagonal elements
       - Produces final similarity scores for pitch and rhythm
    """
    # Get sequences for the specific case
    case_data = df[df['Case'] == case_number].reset_index(drop=True)
    
    if len(case_data) != 2:
        print(f"Error: Case {case_number} does not have exactly 2 songs")
        return
    
    # Get sequences
    seq1_pitch = case_data.loc[0, 'Relative Pitch']
    seq2_pitch = case_data.loc[1, 'Relative Pitch']
    seq1_rhythm = case_data.loc[0, 'Relative Rhythm']
    seq2_rhythm = case_data.loc[1, 'Relative Rhythm']
    
    # Create cost matrices with n-gram lengths
    pitch_distances, pitch_ngram_len = create_cost_matrix(seq1_pitch, seq2_pitch)
    rhythm_distances, rhythm_ngram_len = create_cost_matrix(seq1_rhythm, seq2_rhythm)
    
    # Transform to similarity scores [0,1] using actual n-gram lengths
    pitch_similarities = log_transform_distances(pitch_distances, pitch_ngram_len)
    rhythm_similarities = log_transform_distances(rhythm_distances, rhythm_ngram_len)
    
    # Calculate final scores with consistent decimal places
    pitch_similarity = round(np.mean(np.diagonal(pitch_similarities)), 4)
    rhythm_similarity = round(np.mean(np.diagonal(rhythm_similarities)), 4)
    
    print(f"\nFinal Similarity Scores for {case_number}:")
    print(f"Pitch Similarity: {pitch_similarity}")
    print(f"Rhythm Similarity: {rhythm_similarity}")
    
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
                                   f"Edit Distance Matrix Values ({feature})")
                plot_matrix_as_table(similarities, seq1, seq2, 
                                   f"Similarity Matrix Values ({feature})", 
                                   is_similarity=True,
                                   similarity_score=score)
            
            if viz_type in ['2', '3']:
                plot_heatmap(distances, seq1, seq2,
                           f"Edit Distance Heatmap ({feature})")
                plot_heatmap(similarities, seq1, seq2,
                           f"Similarity Heatmap ({feature})",
                           is_similarity=True,
                           similarity_score=score)

def analyze_all_cases(df, show_plots=False):
    """Analyze all cases and return their similarity scores."""
    cases = sorted(df['Case'].unique())
    results = []
    
    for case in cases:
        case_data = df[df['Case'] == case].reset_index(drop=True)
        
        if len(case_data) != 2:
            print(f"Error: Case {case} does not have exactly 2 songs")
            continue
        
        # Get ruling (formerly category), song titles and sequences
        ruling = case_data.loc[0, 'Ruling']  # Changed from 'Category' to 'Ruling'
        song1_title = case_data.loc[0, 'File Name']
        song2_title = case_data.loc[1, 'File Name']
        seq1_pitch = case_data.loc[0, 'Relative Pitch']
        seq2_pitch = case_data.loc[1, 'Relative Pitch']
        seq1_rhythm = case_data.loc[0, 'Relative Rhythm']
        seq2_rhythm = case_data.loc[1, 'Relative Rhythm']
        
        # Create cost matrices and calculate scores with n-gram lengths
        pitch_matrix, pitch_ngram_len = create_cost_matrix(seq1_pitch, seq2_pitch)
        rhythm_matrix, rhythm_ngram_len = create_cost_matrix(seq1_rhythm, seq2_rhythm)
        
        # Calculate scores with n-gram lengths
        pitch_similarity = round(calculate_similarity_score(
            log_transform_distances(pitch_matrix, pitch_ngram_len)), 4)
        rhythm_similarity = round(calculate_similarity_score(
            log_transform_distances(rhythm_matrix, rhythm_ngram_len)), 4)
        
        # Store results with ruling
        results.append({
            'Case': case,
            'Ruling': ruling,
            'Song 1': song1_title,
            'Song 2': song2_title,
            'Pitch Similarity': pitch_similarity,
            'Rhythm Similarity': rhythm_similarity
        })
        
        # Only show plots if requested
        if show_plots:
            plot_matrix_as_table(pitch_matrix, seq1_pitch, seq2_pitch, 
                           f"Pitch Cost Matrix - {case} ({ruling})\nSimilarity Score: {pitch_similarity}",
                           is_similarity=True)
            plot_matrix_as_table(rhythm_matrix, seq1_rhythm, seq2_rhythm, 
                           f"Rhythm Cost Matrix - {case} ({ruling})\nSimilarity Score: {rhythm_similarity}",
                           is_similarity=True)
    
    return results

def interactive_menu(df):
    """Interactive menu for user interaction."""
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
    """Save similarity analysis results to CSV file."""
    df = pd.DataFrame(results)
    # Add binary ruling column
    df['Binary Ruling'] = (df['Ruling'] == 'Plagiarism').astype(int)
    # Extract case numbers and sort using raw string for regex
    df['Case_Num'] = df['Case'].str.extract(r'(\d+)').astype(int)
    df = df.sort_values('Case_Num')
    df = df.drop('Case_Num', axis=1)
    # Reorder columns to put Binary Ruling after Ruling
    columns = ['Case', 'Ruling', 'Binary Ruling', 'Song 1', 'Song 2', 'Pitch Similarity', 'Rhythm Similarity']
    df = df[columns]
    df.to_csv(output_path, index=False)
    print(f"Similarity report saved to: {output_path}")

def main():
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
