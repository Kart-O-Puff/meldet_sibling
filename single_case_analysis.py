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
    print("\nNote: The n-gram length will be (window size - 1).")
    print("Example: If window size = 5, the n-gram length will be 4 for relative sequences.")
    window_size = int(input("Enter window size (e.g., 5 for n-gram length of 4): "))
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

def extract_features(notes: List[Tuple[int, float]]) -> Tuple[List[int], List[str], List[int]]:
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
            
        # Convert duration to symbolic label
        if duration >= 4.0:
            duration_type = 'whole'
        elif duration >= 3.0:
            duration_type = 'dotted half'
        elif duration >= 2.0:
            duration_type = 'half'
        elif duration >= 1.5:
            duration_type = 'dotted quarter'
        elif duration >= 1.0:
            duration_type = 'quarter'
        elif duration >= 0.75:
            duration_type = 'dotted eighth'
        elif duration >= 0.5:
            duration_type = 'eighth'
        elif duration >= 0.25:
            duration_type = '16th'
        else:
            duration_type = '32nd'
            
        durations.append(duration_type)
    
    return pitches, durations, durations  # Return the durations twice since we don't need mapped_durations anymore

def calculate_relative_pitch(pitches: List[int]) -> List[int]:
    """Calculate relative pitch intervals."""
    if not pitches or len(pitches) < 2:
        return []
    try:
        return [b - a for a, b in zip(pitches[:-1], pitches[1:])]
    except Exception as e:
        print(f"Error calculating relative pitch: {e}")
        return []

def compute_relative_rhythm(durations: List[str]) -> List[float]:
    """Calculate relative rhythm ratios."""
    if not durations or len(durations) < 2:
        return []
    
    duration_map = {
        'whole': 4.0,
        'half': 2.0,
        'quarter': 1.0,
        'eighth': 0.5,
        '16th': 0.25,
        '32nd': 0.125,
        'dotted half': 3.0,
        'dotted quarter': 1.5,
        'dotted eighth': 0.75
    }
    
    try:
        numeric_durations = [duration_map[r] for r in durations]
        relative_rhythm = [round(numeric_durations[i+1] / numeric_durations[i], 3) 
                         for i in range(len(numeric_durations) - 1)]
        return relative_rhythm
    except Exception as e:
        print(f"Error calculating relative rhythm: {e}")
        return []

def generate_ngrams(sequence: List, window_size: int, step_size: int) -> List[List]:
    """Generate n-grams from sequence using window_size-1 as actual n-gram length."""
    ngram_length = window_size - 1
    ngrams = []
    for i in range(0, len(sequence) - ngram_length + 1, step_size):
        ngram = sequence[i:i + ngram_length]
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
    cost_matrix = np.zeros((m, n), dtype=np.int32)
    
    # Get n-gram length (equal to window_size - 1)
    ngram_length = len(seq1[0]) if seq1 else 0
    
    for i in range(m):
        for j in range(n):
            # Count differences between sequences
            differences = sum(1 for x, y in zip(seq1[i], seq2[j]) if x != y)
            cost_matrix[i][j] = int(differences)
            
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
    # Create figure with adaptive size
    fig, (ax_table, ax_colorbar) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 0.2]}, figsize=(16, 10))
    ax_table.axis('tight')
    ax_table.axis('off')
    
    n_rows, n_cols = len(matrix), len(matrix[0])
    base_font_size = min(10, 200 / max(n_rows, n_cols))  # Base font size
    label_font_size = base_font_size * 0.9  # Slightly smaller for labels
    
    # Format labels to use multiple lines if needed
    def format_label(val, prefix):
        label = str(val)
        if len(label) > 15:  # Split into multiple lines if too long
            parts = [label[i:i+15] for i in range(0, len(label), 15)]
            return f"{prefix}:\n" + "\n".join(parts)
        return f"{prefix}:{label}"
    
    col_labels = [format_label(val, f'SB_{i}') for i, val in enumerate(seq2)]
    row_labels = [format_label(val, f'SA_{i}') for i, val in enumerate(seq1)]
    
    if is_similarity:
        cell_text = [[f'{val:.4f}' for val in row] for row in matrix]
    else:
        cell_text = [[f'{int(val)}' for val in row] for row in matrix]
    
    table = ax_table.table(cellText=cell_text,
                          rowLabels=row_labels,
                          colLabels=col_labels,
                          cellLoc='center',
                          loc='center')
    
    # Adjust cell sizes and fonts
    table.auto_set_font_size(False)
    table.set_fontsize(base_font_size)
    
    cell_height = min(0.08, 1.5 / n_rows)
    cell_width = min(0.12, 2.0 / n_cols)
    
    # Increase width for row labels based on content length
    max_row_label_length = max(len(str(label)) for label in row_labels)
    row_label_width = min(0.4, max(0.2, max_row_label_length * 0.015))  # Adjust multiplier as needed
    
    for pos, cell in table._cells.items():
        if pos[1] == -1:  # Row labels
            cell.set_width(row_label_width)
        else:
            cell.set_width(cell_width)
        
        cell.set_height(cell_height)
        cell.PAD = 0.02
        
        if pos[0] == 0 or pos[1] == -1:  # Headers
            cell.set_edgecolor('none')
            if pos[0] == 0:  # Column headers
                cell.set_height(cell_height * 3)  # More height for multiline labels
                cell.get_text().set_rotation(45)
                cell.get_text().set_ha('right')
                cell.get_text().set_va('bottom')
            cell.get_text().set_fontsize(label_font_size)
    
    # Adjust margins with more left padding
    plt.subplots_adjust(left=0.35, right=0.85, top=0.85, bottom=0.2)
    
    if is_similarity:
        norm = plt.Normalize(vmin=0, vmax=1)
        colorbar_label = 'Similarity Value\n(0: Most Different, 1: Most Similar)'
        cmap = plt.cm.viridis
    else:
        max_edit_distance = ngram_length
        norm = plt.Normalize(vmin=0, vmax=max_edit_distance)
        colorbar_label = f'Edit Distance\n(0: Most Similar, {max_edit_distance}: Most Different)'
        cmap = plt.cm.viridis_r
    
    # Color cells and mark diagonal
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
    
    # Adjust layout
    plt.subplots_adjust(left=0.2, top=0.85, bottom=0.2)
    
    cmap_obj = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(cmap_obj, cax=ax_colorbar, label=colorbar_label)
    
    full_title = f"{title}\n"
    if song1 and song2:
        full_title += f"Song A: {song1} vs Song B: {song2}\n"
    if similarity_score is not None:
        interpretation = interpret_similarity(similarity_score * 100)
        full_title += f"Similarity Score: {similarity_score * 100:.2f}% - {interpretation}"
    title_obj = ax_table.set_title(full_title, pad=50, y=1.5)
    title_obj.set_fontsize(14)  # Increased title font size
    
    plt.tight_layout()
    plt.show()

def plot_heatmap(matrix, seq1, seq2, title, is_similarity=False, similarity_score=None, song1="", song2="", forced_diagonal=None, ngram_length=None):
    """Plot similarity/distance matrix as a heatmap."""
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 12})
    
    # Format sequences for display
    if isinstance(seq1[0], list):  # For n-grams
        seq1_labels = [str(s) for s in seq1]
        seq2_labels = [str(s) for s in seq2]
    else:  # For single values
        seq1_labels = [str(round(float(s), 3)) if isinstance(s, (float, str)) else str(s) for s in seq1]
        seq2_labels = [str(round(float(s), 3)) if isinstance(s, (float, str)) else str(s) for s in seq2]
    
    if is_similarity:
        vmin, vmax = 0, 1
        cmap = 'viridis'
        cbar_label = 'Similarity Value\n(0: Most Different, 1: Most Similar)'
    else:
        max_edit_distance = ngram_length
        vmin, vmax = 0, max_edit_distance
        cmap = 'viridis_r'
        cbar_label = f'Edit Distance\n(0: Most Similar, {max_edit_distance}: Most Different)'
    
    im = plt.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im)
    cbar.set_label(cbar_label, fontsize=12)
    
    plt.xlabel(f"Song B: {song2}", fontsize=12)
    plt.ylabel(f"Song A: {song1}", fontsize=12)
    
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
        plt.legend(fontsize=12)
    
    full_title = f"{title}\n"
    if song1 and song2:
        full_title += f"{song1} vs {song2}\n"
    if similarity_score is not None:
        interpretation = interpret_similarity(similarity_score * 100)
        full_title += f"Similarity Score: {similarity_score * 100:.2f}% - {interpretation}"
    plt.title(full_title, pad=20, fontsize=14)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function for single case analysis."""
    # Get single case analysis folder path
    base_path = Path(__file__).parent / "Single_Case_Analysis"
    
    # Ensure 1v1_Reports folder exists
    reports_folder = base_path / "1v1_Reports"
    reports_folder.mkdir(exist_ok=True)
    
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
    pitches1, durations1, mapped_durations1 = extract_features(notes1)
    pitches2, durations2, mapped_durations2 = extract_features(notes2)
    
    # Calculate relative sequences
    relative_pitch1 = calculate_relative_pitch(pitches1)
    relative_pitch2 = calculate_relative_pitch(pitches2)
    relative_rhythm1 = compute_relative_rhythm(durations1)
    relative_rhythm2 = compute_relative_rhythm(durations2)
    
    # Generate n-grams for relative sequences
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
        'Absolute_Pitch_File1': str(pitches1),
        'Absolute_Pitch_File2': str(pitches2),
        'Absolute_Rhythm_File1': str(durations1),
        'Absolute_Rhythm_File2': str(durations2),
        'Relative_Pitch_File1': str(relative_pitch1),
        'Relative_Pitch_File2': str(relative_pitch2),
        'Relative_Rhythm_File1': str([round(x, 3) for x in relative_rhythm1]),
        'Relative_Rhythm_File2': str([round(x, 3) for x in relative_rhythm2]),
        'Segmented_Pitch_Ngrams_File1': str(pitch_ngrams1),
        'Segmented_Pitch_Ngrams_File2': str(pitch_ngrams2),
        'Segmented_Rhythm_Ngrams_File1': str(rhythm_ngrams1),
        'Segmented_Rhythm_Ngrams_File2': str(rhythm_ngrams2),
        'Pitch_Similarity': f"{pitch_similarity:.2f}%",
        'Pitch_Interpretation': pitch_interpretation,
        'Rhythm_Similarity': f"{rhythm_similarity:.2f}%",
        'Rhythm_Interpretation': rhythm_interpretation
    }
    
    # Save results to CSV and set output path
    df = pd.DataFrame([results])
    output_path = reports_folder / f"similarity_{file1.stem}_and_{file2.stem}.csv"
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