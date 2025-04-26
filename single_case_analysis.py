import os
import pandas as pd
import numpy as np
from music21 import *
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def get_user_input() -> Tuple[int, int]:
    """Get window size and step size from user."""
    window_size = int(input("Enter window size (e.g., 3): "))
    step_size = int(input("Enter step size (e.g., 1): "))
    return window_size, step_size

def load_midi_file(file_path: str) -> List[Tuple[int, float]]:
    """Load and extract notes from MIDI file."""
    try:
        # Load MIDI file
        midi_stream = converter.parse(file_path)
        notes = []
        
        # Extract notes and rests
        for element in midi_stream.recurse():
            if isinstance(element, note.Note):
                notes.append((element.pitch.midi, element.duration.quarterLength))
            elif isinstance(element, chord.Chord):
                notes.append((element.sortedCommonName, element.duration.quarterLength))
        
        return notes
    except Exception as e:
        print(f"Error loading MIDI file {file_path}: {str(e)}")
        return []

def extract_features(notes: List[Tuple[int, float]]) -> Tuple[List[int], List[float]]:
    """Extract pitch and rhythm features from notes."""
    pitches = []
    durations = []
    
    for pitch, duration in notes:
        if isinstance(pitch, int):
            pitches.append(pitch)
        elif isinstance(pitch, str):
            # For chords, use the highest note
            chord_notes = [int(note) for note in pitch.split()]
            pitches.append(max(chord_notes))
        durations.append(duration)
    
    return pitches, durations

def calculate_relative_pitch(pitches: List[int]) -> List[int]:
    """Calculate relative pitch intervals."""
    if not pitches or len(pitches) < 2:
        return []
    try:
        return [b - a for a, b in zip(pitches[:-1], pitches[1:])]
    except Exception as e:
        print(f"Error calculating relative pitch: {e}")
        return []

def calculate_relative_rhythm(durations: List[float]) -> List[float]:
    """Calculate relative rhythm ratios."""
    if not durations or len(durations) < 2:
        return []
    try:
        return [round(b/a, 3) if a != 0 else 0 for a, b in zip(durations[:-1], durations[1:])]
    except Exception as e:
        print(f"Error calculating relative rhythm: {e}")
        return []

def generate_ngrams(sequence: List, window_size: int, step_size: int) -> List[List]:
    """Generate n-grams from sequence."""
    ngrams = []
    for i in range(0, len(sequence) - window_size + 1, step_size):
        ngram = sequence[i:i + window_size]
        ngrams.append(ngram)
    return ngrams

def calculate_log_transform_distance(value, max_d):
    """Transform edit distance to similarity score using logarithmic scaling."""
    try:
        log_similarity = 1 - (np.log2(1 + value) / np.log2(1 + max_d))
        return round(max(0.0, min(1.0, log_similarity)), 4)
    except:
        return 0.0

def log_transform_distances(distance_matrix, ngram_length):
    """Convert distance matrix to similarity scores."""
    similarity_matrix = np.zeros_like(distance_matrix, dtype=float)
    max_distance = ngram_length
    
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            distance = distance_matrix[i, j]
            similarity_matrix[i, j] = calculate_log_transform_distance(distance, max_distance)
    
    return similarity_matrix

def calculate_similarity_score(matrix, forced_shift=None):
    """Calculate similarity score using best diagonal alignment."""
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

    if forced_shift is not None:
        return scores[forced_shift]
    return max(scores)

def create_cost_matrix(seq1: List[List], seq2: List[List]) -> Tuple[np.ndarray, int]:
    """Create cost matrix and calculate edit distances between sequences."""
    m, n = len(seq1), len(seq2)
    cost_matrix = np.zeros((m, n), dtype=np.int32)  # Changed to integer type
    
    # Get n-gram length
    ngram_length = len(seq1[0]) if seq1 else 0
    
    for i in range(m):
        for j in range(n):
            # Count differences between sequences
            differences = sum(1 for x, y in zip(seq1[i], seq2[j]) if x != y)
            cost_matrix[i][j] = int(differences)  # Ensure integer values
            
    return cost_matrix, ngram_length

def interpret_similarity(similarity_score: float) -> str:
    """Interpret similarity score."""
    if similarity_score >= 75:
        return "Very High Similarity"
    elif similarity_score >= 50:
        return "High Similarity"
    elif similarity_score >= 25:
        return "Moderate Similarity"
    else:
        return "Low Similarity"

def plot_matrix_as_table(matrix, seq1, seq2, title, is_similarity=False, similarity_score=None, song1="", song2="", forced_diagonal=None, ngram_length=None):
    """Visualize similarity/distance matrix with detailed values."""
    fig, (ax_table, ax_colorbar) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 0.2]}, figsize=(14, 8))
    ax_table.axis('tight')
    ax_table.axis('off')
    
    col_labels = [f'S2_{i}:{val}' for i, val in enumerate(seq2)]
    row_labels = [f'S1_{i}:{val}' for i, val in enumerate(seq1)]
    
    if is_similarity:
        cell_text = [[f'{val:.4f}' for val in row] for row in matrix]
    else:
        cell_text = [[f'{int(val)}' for val in row] for row in matrix]
    
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
        max_edit_distance = ngram_length if ngram_length else 7
        norm = plt.Normalize(vmin=0, vmax=max_edit_distance)
        colorbar_label = f'Edit Distance\n(0: Most Similar, {max_edit_distance}: Most Different)'
        cmap = plt.cm.viridis_r
    
    # Color cells based on values
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            cell = table[i+1, j]
            value = matrix[i, j]
            color = cmap(norm(value))
            cell.set_facecolor(color)
            
            if forced_diagonal is not None:
                m, n = matrix.shape
                min_len = min(m, n)
                if m <= n and i < min_len and j == i + forced_diagonal:
                    cell.set_edgecolor('red')
                    cell.set_linewidth(2)
                elif m > n and j < min_len and i == j + forced_diagonal:
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

def plot_heatmap(matrix, seq1, seq2, title, is_similarity=False, similarity_score=None, song1="", song2="", forced_diagonal=None, ngram_length=None):
    """Plot similarity/distance matrix as a heatmap."""
    plt.figure(figsize=(10, 8))
    
    if is_similarity:
        vmin, vmax = 0, 1
        cmap = 'viridis'
        cbar_label = 'Similarity Value\n(0: Most Different, 1: Most Similar)'
    else:
        max_edit_distance = ngram_length if ngram_length else 7
        vmin, vmax = 0, max_edit_distance
        cmap = 'viridis_r'
        cbar_label = f'Edit Distance\n(0: Most Similar, {max_edit_distance}: Most Different)'
    
    im = plt.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=cbar_label)
    plt.xlabel(f"Song B: {song2}")
    plt.ylabel(f"Song A: {song1}")
    
    plt.xticks(range(len(seq2)), [f'S2_{i}' for i in range(len(seq2))])
    plt.yticks(range(len(seq1)), [f'S1_{i}' for i in range(len(seq1))])
    
    if forced_diagonal is not None:
        m, n = matrix.shape
        min_len = min(m, n)
        if m <= n:
            plt.plot(range(forced_diagonal, forced_diagonal + min_len),
                    range(min_len),
                    'rx-', markersize=8, linewidth=2, label='Best diagonal')
        else:
            plt.plot(range(min_len),
                    range(forced_diagonal, forced_diagonal + min_len),
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

def main():
    """Main function for single case analysis."""
    # Get single case analysis folder path
    base_path = Path(__file__).parent / "Single_Case_Analysis"
    
    # List MIDI files
    midi_files = list(base_path.glob("*.mid"))
    if len(midi_files) < 2:
        print("Please ensure there are at least 2 MIDI files in the Single_Case_Analysis folder.")
        return
    
    # Show available files and get user selection
    print("\nAvailable MIDI files:")
    for i, file in enumerate(midi_files):
        print(f"{i+1}. {file.name}")
    
    try:
        idx1 = int(input("\nSelect first file number: ")) - 1
        idx2 = int(input("Select second file number: ")) - 1
        
        file1 = midi_files[idx1]
        file2 = midi_files[idx2]
    except (IndexError, ValueError):
        print("Invalid file selection.")
        return
    
    # Get window and step size
    window_size, step_size = get_user_input()
    
    # Process MIDI files
    notes1 = load_midi_file(str(file1))
    notes2 = load_midi_file(str(file2))
    
    # Extract features
    pitches1, durations1 = extract_features(notes1)
    pitches2, durations2 = extract_features(notes2)
    
    # Calculate relative sequences
    relative_pitch1 = calculate_relative_pitch(pitches1)
    relative_pitch2 = calculate_relative_pitch(pitches2)
    relative_rhythm1 = calculate_relative_rhythm(durations1)
    relative_rhythm2 = calculate_relative_rhythm(durations2)
    
    # Generate n-grams from relative sequences
    pitch_ngrams1 = generate_ngrams(relative_pitch1, window_size, step_size)
    pitch_ngrams2 = generate_ngrams(relative_pitch2, window_size, step_size)
    rhythm_ngrams1 = generate_ngrams(relative_rhythm1, window_size, step_size)
    rhythm_ngrams2 = generate_ngrams(relative_rhythm2, window_size, step_size)
    
    # Generate cost matrices and calculate similarities
    pitch_cost_matrix, pitch_ngram_len = create_cost_matrix(pitch_ngrams1, pitch_ngrams2)
    rhythm_cost_matrix, rhythm_ngram_len = create_cost_matrix(rhythm_ngrams1, rhythm_ngrams2)
    
    # Transform distances to similarities
    pitch_similarities = log_transform_distances(pitch_cost_matrix, pitch_ngram_len)
    rhythm_similarities = log_transform_distances(rhythm_cost_matrix, rhythm_ngram_len)
    
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
    pitch_similarity = calculate_similarity_score(pitch_similarities, forced_shift=best_shift) * 100
    rhythm_similarity = calculate_similarity_score(rhythm_similarities, forced_shift=best_shift) * 100
    
    # Generate interpretations
    pitch_interpretation = interpret_similarity(pitch_similarity)
    rhythm_interpretation = interpret_similarity(rhythm_similarity)
    
    # Prepare results and initialize output path
    results = {
        'File1': file1.name,
        'File2': file2.name,
        'Window_Size': window_size,
        'Step_Size': step_size,
        'Absolute_Pitch_Ngrams_File1': str([list(map(int, pitches1))]),
        'Absolute_Pitch_Ngrams_File2': str([list(map(int, pitches2))]),
        'Absolute_Rhythm_Ngrams_File1': str([list(map(float, durations1))]),
        'Absolute_Rhythm_Ngrams_File2': str([list(map(float, durations2))]),
        'Relative_Pitch_File1': str(relative_pitch1),
        'Relative_Pitch_File2': str(relative_pitch2),
        'Relative_Rhythm_File1': str(relative_rhythm1),
        'Relative_Rhythm_File2': str(relative_rhythm2),
        'Pitch_Similarity': f"{pitch_similarity:.2f}%",
        'Pitch_Interpretation': pitch_interpretation,
        'Rhythm_Similarity': f"{rhythm_similarity:.2f}%",
        'Rhythm_Interpretation': rhythm_interpretation
    }
    
    # Save results to CSV and set output path
    df = pd.DataFrame([results])
    output_path = base_path / f"similarity_{file1.stem}_and_{file2.stem}.csv"
    df.to_csv(output_path, index=False)
    
    # Display results menu
    while True:
        print("\nVisualization Options:")
        print("1. Matrix Tables")
        print("   a) Edit Distance Matrix Values")
        print("   b) Similarity Matrix Values")
        print("   c) Both Matrix Tables")
        print("2. Similarity Heatmaps")
        print("3. Show All")
        print("4. Exit")
        
        choice = input("\nSelect visualization option (1a, 1b, 1c, 2, 3, 4): ")
        
        if choice == '1a':
            # Show edit distance matrices
            plot_matrix_as_table(pitch_cost_matrix, pitch_ngrams1, pitch_ngrams2,
                               "Pitch Edit Distance Matrix",
                               is_similarity=False,
                               song1=file1.name, song2=file2.name,
                               forced_diagonal=best_shift,
                               ngram_length=pitch_ngram_len)
            
            plot_matrix_as_table(rhythm_cost_matrix, rhythm_ngrams1, rhythm_ngrams2,
                               "Rhythm Edit Distance Matrix",
                               is_similarity=False,
                               song1=file1.name, song2=file2.name,
                               forced_diagonal=best_shift,
                               ngram_length=rhythm_ngram_len)
        
        elif choice == '1b':
            # Show similarity matrices
            plot_matrix_as_table(pitch_similarities, pitch_ngrams1, pitch_ngrams2,
                               "Pitch Similarity Matrix",
                               is_similarity=True,
                               similarity_score=pitch_similarity/100,
                               song1=file1.name, song2=file2.name,
                               forced_diagonal=best_shift)
            
            plot_matrix_as_table(rhythm_similarities, rhythm_ngrams1, rhythm_ngrams2,
                               "Rhythm Similarity Matrix",
                               is_similarity=True,
                               similarity_score=rhythm_similarity/100,
                               song1=file1.name, song2=file2.name,
                               forced_diagonal=best_shift)
        
        elif choice == '1c':
            # Show both matrix types
            print("\nEdit Distance Matrices:")
            plot_matrix_as_table(pitch_cost_matrix, pitch_ngrams1, pitch_ngrams2,
                               "Pitch Edit Distance Matrix",
                               is_similarity=False,
                               song1=file1.name, song2=file2.name,
                               forced_diagonal=best_shift,
                               ngram_length=pitch_ngram_len)
            
            plot_matrix_as_table(rhythm_cost_matrix, rhythm_ngrams1, rhythm_ngrams2,
                               "Rhythm Edit Distance Matrix",
                               is_similarity=False,
                               song1=file1.name, song2=file2.name,
                               forced_diagonal=best_shift,
                               ngram_length=rhythm_ngram_len)
            
            print("\nSimilarity Matrices:")
            plot_matrix_as_table(pitch_similarities, pitch_ngrams1, pitch_ngrams2,
                               "Pitch Similarity Matrix",
                               is_similarity=True,
                               similarity_score=pitch_similarity/100,
                               song1=file1.name, song2=file2.name,
                               forced_diagonal=best_shift)
            
            plot_matrix_as_table(rhythm_similarities, rhythm_ngrams1, rhythm_ngrams2,
                               "Rhythm Similarity Matrix",
                               is_similarity=True,
                               similarity_score=rhythm_similarity/100,
                               song1=file1.name, song2=file2.name,
                               forced_diagonal=best_shift)
        
        elif choice == '2':
            # Show similarity heatmaps
            plot_heatmap(pitch_similarities, pitch_ngrams1, pitch_ngrams2,
                        "Pitch Similarity Heatmap",
                        is_similarity=True,
                        similarity_score=pitch_similarity/100,
                        song1=file1.name, song2=file2.name,
                        forced_diagonal=best_shift)
            
            plot_heatmap(rhythm_similarities, rhythm_ngrams1, rhythm_ngrams2,
                        "Rhythm Similarity Heatmap",
                        is_similarity=True,
                        similarity_score=rhythm_similarity/100,
                        song1=file1.name, song2=file2.name,
                        forced_diagonal=best_shift)
        
        elif choice == '3':
            # Show all visualizations
            print("\nEdit Distance Matrices:")
            plot_matrix_as_table(pitch_cost_matrix, pitch_ngrams1, pitch_ngrams2,
                               "Pitch Edit Distance Matrix",
                               is_similarity=False,
                               song1=file1.name, song2=file2.name,
                               forced_diagonal=best_shift,
                               ngram_length=pitch_ngram_len)
            
            plot_matrix_as_table(rhythm_cost_matrix, rhythm_ngrams1, rhythm_ngrams2,
                               "Rhythm Edit Distance Matrix",
                               is_similarity=False,
                               song1=file1.name, song2=file2.name,
                               forced_diagonal=best_shift,
                               ngram_length=rhythm_ngram_len)
            
            print("\nSimilarity Matrices and Heatmaps:")
            plot_matrix_as_table(pitch_similarities, pitch_ngrams1, pitch_ngrams2,
                               "Pitch Similarity Matrix",
                               is_similarity=True,
                               similarity_score=pitch_similarity/100,
                               song1=file1.name, song2=file2.name,
                               forced_diagonal=best_shift)
            
            plot_matrix_as_table(rhythm_similarities, rhythm_ngrams1, rhythm_ngrams2,
                               "Rhythm Similarity Matrix",
                               is_similarity=True,
                               similarity_score=rhythm_similarity/100,
                               song1=file1.name, song2=file2.name,
                               forced_diagonal=best_shift)
            
            print("\nSimilarity Heatmaps:")
            plot_heatmap(pitch_similarities, pitch_ngrams1, pitch_ngrams2,
                        "Pitch Similarity Heatmap",
                        is_similarity=True,
                        similarity_score=pitch_similarity/100,
                        song1=file1.name, song2=file2.name,
                        forced_diagonal=best_shift)
            
            plot_heatmap(rhythm_similarities, rhythm_ngrams1, rhythm_ngrams2,
                        "Rhythm Similarity Heatmap",
                        is_similarity=True,
                        similarity_score=rhythm_similarity/100,
                        song1=file1.name, song2=file2.name,
                        forced_diagonal=best_shift)
        
        elif choice == '4':
            print(f"\nFinal Similarity Analysis Results for '{file1.name}' and '{file2.name}':")
            print(f"Pitch Similarity: {pitch_similarity:.2f}% - {pitch_interpretation}")
            print(f"Rhythm Similarity: {rhythm_similarity:.2f}% - {rhythm_interpretation}")
            print(f"\nResults saved to: {output_path}")
            break
        else:
            print("Invalid choice. Please try again.")
    
    # Display results
    print(f"\nSimilarity Scores for '{file1.name}' and '{file2.name}':")
    print(f"Pitch Similarity Score: {pitch_similarity:.2f}% - {pitch_interpretation}")
    print(f"Rhythm Similarity Score: {rhythm_similarity:.2f}% - {rhythm_interpretation}")
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()