import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ast

def load_sequences_from_library():
    """Load and parse the melody library CSV file."""
    csv_path = Path(__file__).parent / "MCIC_Dataset" / "MCIC_Preprocessed" / "melody_library_symbolic.csv"
    df = pd.read_csv(csv_path, encoding='cp1252')
    
    # Convert string representations back to actual lists
    df['Relative Pitch'] = df['Relative Pitch'].apply(ast.literal_eval)
    df['Relative Rhythm'] = df['Relative Rhythm'].apply(ast.literal_eval)
    
    # Ensure 'Ruling' column exists
    if 'Category' in df.columns and 'Ruling' not in df.columns:
        df['Ruling'] = df['Category']
        df = df.drop('Category', axis=1)
    
    return df

def calculate_common_elements(seq1, seq2):
    """
    Calculate Jaccard similarity between two songs based on their n-gram sequences.
    
    Similarity = |A ∩ B| / (|A| + |B| - |A ∩ B|)
    where:
    - A = set of n-grams in song1
    - B = set of n-grams in song2
    - |A ∩ B| = number of common n-grams (exact matches)
    - |A| = total n-grams in song1
    - |B| = total n-grams in song2
    """
    try:
        # Input validation
        if not isinstance(seq1, (list, np.ndarray)) or not isinstance(seq2, (list, np.ndarray)):
            return 0.0
            
        # Convert n-gram sequences to tuples for set operations
        seq1_units = set(tuple(gram) for gram in seq1 if gram is not None)
        seq2_units = set(tuple(gram) for gram in seq2 if gram is not None)
        
        if not seq1_units or not seq2_units:
            return 0.0
            
        # Calculate intersection (common n-grams)
        common_ngrams = seq1_units.intersection(seq2_units)
        
        # Calculate similarity using the formula
        # similarity = |A ∩ B| / (|A| + |B| - |A ∩ B|)
        intersection_size = len(common_ngrams)
        similarity = intersection_size / (len(seq1_units) + len(seq2_units) - intersection_size)
        
        return round(similarity, 4)
        
    except Exception as e:
        return 0.0

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

def plot_similarity_comparison(similarities, title, song1, song2):
    """Plot similarity scores for complete n-gram sequence comparison."""
    plt.figure(figsize=(14, 6))  # Increased figure width to accommodate legend
    
    # Convert to percentage for display
    similarities_pct = similarities * 100
    
    # Single bar showing Jaccard similarity
    plt.bar(['Jaccard Similarity'], [similarities_pct], color='skyblue', alpha=0.6)
    
    # Add score label
    plt.text(0, similarities_pct + 2, f'{similarities_pct:.2f}%', ha='center')
    
    plt.title(f"{title}\n{song1} vs {song2}")
    plt.ylabel("Similarity Score (%)")
    plt.ylim(0, 115)
    plt.grid(True, alpha=0.3)
    
    # Add info box - moved to the right side
    info_text = (
        f'Sum Common Similarity Score: {similarities_pct:.2f}% ({interpret_similarity(similarities_pct)})\n'
        f'Method: Complete N-gram Sequence Comparison\n'
        f'(Exact matches of entire n-gram sequences)'
    )
    plt.text(1.02, 0.98, info_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top',
             fontsize=9)
    
    plt.tight_layout()
    plt.show()

def plot_ngram_analysis(seq1, seq2, title, song1, song2, similarity_score):
    """Plot n-gram sequence analysis showing exact sequence matches."""
    # Convert complete n-gram sequences to tuples for comparison
    seq1_units = set(tuple(gram) for gram in seq1 if gram is not None)
    seq2_units = set(tuple(gram) for gram in seq2 if gram is not None)
    
    # Calculate set operations on complete sequences
    common = len(seq1_units.intersection(seq2_units))
    unique_to_s1 = len(seq1_units.difference(seq2_units))
    unique_to_s2 = len(seq2_units.difference(seq1_units))
    
    # Create stacked bar chart with increased figure width
    plt.figure(figsize=(14, 6))
    
    songs = ['Song A', 'Song B']
    common_values = [common, common]
    unique_values = [unique_to_s1, unique_to_s2]
    
    bar_width = 0.35
    plt.bar(songs, common_values, bar_width, color='green', alpha=0.6)
    plt.bar(songs, unique_values, bar_width, 
            bottom=common_values, color=['blue', 'red'], alpha=0.6)
    
    # Add value labels
    for i in range(len(songs)):
        plt.text(i, common_values[i]/2, 
                f'Common: {common_values[i]}', 
                ha='center', va='center')
        plt.text(i, common_values[i] + unique_values[i]/2,
                f'Unique: {unique_values[i]}',
                ha='center', va='center')
        plt.text(i, common_values[i] + unique_values[i],
                f'Total: {common_values[i] + unique_values[i]}',
                ha='center', va='bottom')
    
    plt.title(f"{title} N-gram Sequence Analysis\n{song1} vs {song2}\n"
              f"Sum Common Similarity Score: {similarity_score * 100:.2f}% ({interpret_similarity(similarity_score * 100)})")
    plt.ylabel('Number of N-gram Sequences')
    plt.grid(True, alpha=0.3)
    
    # Add info box on the right side
    info_text = (
        f'Common N-gram Sequences: {common}\n'
        f'Unique to Song A: {unique_to_s1}\n'
        f'Unique to Song B: {unique_to_s2}\n'
        f'Total Sequences in Song A: {len(seq1_units)}\n'
        f'Total Sequences in Song B: {len(seq2_units)}\n'
        f'Sum Common Similarity Score = ({common}/{len(seq1_units) + len(seq2_units) - common})'
        f' = {similarity_score * 100:.2f}% ({interpret_similarity(similarity_score * 100)})'
    )
    plt.text(1.02, 0.98, info_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='bottom',
             fontsize=9)
    
    plt.tight_layout()
    plt.show()

def analyze_case(df, case_number):
    """
    Analyze a single case using Sum Common approach (Jaccard Similarity).
    
    Important Note on Similarity Calculation:
    1. The calculate_common_elements() function implements true Jaccard similarity:
       - Treats all n-grams as a single set per sequence
       - Calculates |intersection| / |union| once per feature
       - Returns a single similarity score [0,1]
    
    2. This is different from the visualization function which:
       - Shows individual n-gram pair comparisons
       - Displays an average line (not used for actual similarity)
       - Is only for visualization purposes
    
    The similarity score used in the final report is the true Jaccard similarity
    from calculate_common_elements(), not the average from visualizations.
    """
    case_data = df[df['Case'] == case_number].reset_index(drop=True)
    
    if len(case_data) != 2:
        print(f"Error: Case {case_number} does not have exactly 2 songs")
        return None
    
    # Get sequences (changed back to relative intervals)
    seq1_pitch = case_data.loc[0, 'Relative Pitch']
    seq2_pitch = case_data.loc[1, 'Relative Pitch']
    seq1_rhythm = case_data.loc[0, 'Relative Rhythm']
    seq2_rhythm = case_data.loc[1, 'Relative Rhythm']
    
    # Calculate single similarity score for each feature
    pitch_score = calculate_common_elements(seq1_pitch, seq2_pitch)
    rhythm_score = calculate_common_elements(seq1_rhythm, seq2_rhythm)
    
    return {
        'Case': case_number,
        'Ruling': case_data.loc[0, 'Ruling'],
        'Song A': case_data.loc[0, 'File Name'],
        'Song B': case_data.loc[1, 'File Name'],
        'Pitch Similarity': pitch_score,
        'Rhythm Similarity': rhythm_score
    }

def analyze_case_with_visualization(df, case_number):
    """Analyze a single case with visualization options."""
    case_data = df[df['Case'] == case_number].reset_index(drop=True)
    
    if len(case_data) != 2:
        print(f"Error: Case {case_number} does not have exactly 2 songs")
        return None
    
    # Get sequences
    seq1_pitch = case_data.loc[0, 'Relative Pitch']
    seq2_pitch = case_data.loc[1, 'Relative Pitch']
    seq1_rhythm = case_data.loc[0, 'Relative Rhythm']
    seq2_rhythm = case_data.loc[1, 'Relative Rhythm']
    
    # Calculate single document-level similarity score
    pitch_score = calculate_common_elements(seq1_pitch, seq2_pitch)
    rhythm_score = calculate_common_elements(seq1_rhythm, seq2_rhythm)
    
    print(f"\nFinal Similarity Scores for {case_number}:")
    print(f"Pitch Similarity: {pitch_score * 100:.2f}% ({interpret_similarity(pitch_score * 100)})")
    print(f"Rhythm Similarity: {rhythm_score * 100:.2f}% ({interpret_similarity(rhythm_score * 100)})")
    
    print("\nVisualization Options:")
    print("1. Show Pitch Analysis")
    print("2. Show Rhythm Analysis")
    print("3. Show Both Analyses")
    viz_choice = input("Enter your choice (1-3): ")
    
    if viz_choice in ['1', '3']:
        plot_ngram_analysis(
            seq1_pitch, seq2_pitch,
            "Pitch Feature",
            case_data.loc[0, 'File Name'],
            case_data.loc[1, 'File Name'],
            pitch_score
        )
    
    if viz_choice in ['2', '3']:
        plot_ngram_analysis(
            seq1_rhythm, seq2_rhythm,
            "Rhythm Feature",
            case_data.loc[0, 'File Name'],
            case_data.loc[1, 'File Name'],
            rhythm_score
        )
    
    return {
        'Case': case_number,
        'Ruling': case_data.loc[0, 'Ruling'],
        'Song A': case_data.loc[0, 'File Name'],
        'Song B': case_data.loc[1, 'File Name'],
        'Pitch Similarity': pitch_score,
        'Rhythm Similarity': rhythm_score
    }

def analyze_all_cases(df):
    """Analyze all cases and return their similarity scores."""
    cases = sorted(df['Case'].unique())
    results = []
    
    for case in cases:
        result = analyze_case(df, case)
        if result:
            results.append(result)
    
    return results

def save_similarity_report(results, output_path):
    """Save similarity analysis results to CSV file with percentage values."""
    df = pd.DataFrame(results)
    
    # Convert similarity scores to percentages
    df['Pitch Similarity'] = df['Pitch Similarity'] * 100
    df['Rhythm Similarity'] = df['Rhythm Similarity'] * 100
    
    # Add interpretation columns
    df['Pitch Interpretation'] = df['Pitch Similarity'].apply(interpret_similarity)
    df['Rhythm Interpretation'] = df['Rhythm Similarity'].apply(interpret_similarity)
    
    # Add binary ruling column
    df['Binary Ruling'] = (df['Ruling'] == 'Plagiarism').astype(int)
    
    # Extract case numbers and sort using raw string
    df['Case_Num'] = df['Case'].str.extract(r'(\d+)').astype(int)
    df = df.sort_values('Case_Num')
    df = df.drop('Case_Num', axis=1)
    
    # Reorder columns
    columns = ['Case', 'Ruling', 'Binary Ruling', 'Song A', 'Song B', 
              'Pitch Similarity', 'Pitch Interpretation', 
              'Rhythm Similarity', 'Rhythm Interpretation']
    df = df[columns]
    df.to_csv(output_path, index=False)
    print(f"Similarity report saved to: {output_path}")

def interactive_menu(df):
    """Interactive menu for user interaction."""
    while True:
        print("\nSum Common Analysis Options:")
        print("1. Show case analysis (with visualizations)")
        print("2. Show case analysis (without visualizations)")
        print("3. List available cases")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            case = input("Enter case number (e.g., Case_006): ")
            if case in df['Case'].unique():
                analyze_case_with_visualization(df, case)
            else:
                print("Invalid case number!")
        elif choice == '2':
            case = input("Enter case number (e.g., Case_006): ")
            if case in df['Case'].unique():
                result = analyze_case(df, case)
                if result:
                    print(f"\nResults for {case}:")
                    print(f"Pitch Similarity: {result['Pitch Similarity'] * 100:.2f}%")
                    print(f"Rhythm Similarity: {result['Rhythm Similarity'] * 100:.2f}%")
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
    """
    Key Points of Sum Common (Jaccard) Approach:
    1. Treats each n-gram sequence as a complete unit
    2. Uses set operations to find common sequences
    3. Calculates similarity using Jaccard formula:
       |A ∩ B| / (|A| + |B| - |A ∩ B|)
    """
    # Load sequences from library
    df = load_sequences_from_library()
    
    # Analyze all cases without printing updates
    print("Analyzing all cases using Sum Common approach...")
    results = analyze_all_cases(df)
    
    # Save results to CSV
    report_path = Path(__file__).parent / "similarity_report_sumcommon.csv"
    save_similarity_report(results, report_path)
    
    print(f"\nAnalysis completed. Results saved to {report_path}")
    
    # Start interactive menu
    interactive_menu(df)

if __name__ == "__main__":
    main()
