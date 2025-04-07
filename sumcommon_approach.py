import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ast

def load_sequences_from_library():
    """Load and parse the melody library CSV file."""
    csv_path = Path(__file__).parent / "MCIC_Dataset" / "MCIC_Preprocessed" / "melody_library.csv"
    df = pd.read_csv(csv_path)
    
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
    
    Error Handling:
    - Checks if input sequences are valid lists/arrays
    - Ensures input sequences are not empty
    - Converts non-list tokens to lists if possible
    - Returns 0.0 for any error cases
    """
    try:
        # Input validation
        if not isinstance(seq1, (list, np.ndarray)) or not isinstance(seq2, (list, np.ndarray)):
            return 0.0
            
        # Convert sequences to sets of tuples, handling potential non-iterable elements
        seq1_units = set()
        seq2_units = set()
        
        for unit in seq1:
            if unit is not None:
                try:
                    # Try to convert to tuple, handle both lists and single values
                    if isinstance(unit, (list, np.ndarray)):
                        seq1_units.add(tuple(unit))
                    else:
                        seq1_units.add((unit,))
                except TypeError:
                    continue  # Skip invalid units
                    
        for unit in seq2:
            if unit is not None:
                try:
                    if isinstance(unit, (list, np.ndarray)):
                        seq2_units.add(tuple(unit))
                    else:
                        seq2_units.add((unit,))
                except TypeError:
                    continue  # Skip invalid units
        
        # Check if we have valid units after conversion
        if not seq1_units or not seq2_units:
            return 0.0
            
        # Calculate Jaccard similarity
        common_ngrams = seq1_units.intersection(seq2_units)
        all_unique_ngrams = seq1_units.union(seq2_units)
        
        if not all_unique_ngrams:
            return 0.0
            
        return len(common_ngrams) / len(all_unique_ngrams)
        
    except Exception as e:
        return 0.0

def plot_similarity_comparison(similarities, title, song1, song2):
    """
    Plot similarity scores for corresponding n-gram pairs and their average.
    Each bar represents a single n-gram pair comparison.
    """
    plt.figure(figsize=(12, 6))
    
    # Create bar plot for n-gram similarities
    x = range(len(similarities))
    plt.bar(x, similarities, color='skyblue', alpha=0.6, label='N-gram Pair Similarity')
    
    # Add average line
    avg_similarity = np.mean(similarities)
    plt.axhline(y=avg_similarity, color='red', linestyle='--', 
                label=f'Average Similarity: {avg_similarity:.4f}')
    
    # Add individual score labels on bars
    for i, score in enumerate(similarities):
        plt.text(i, score + 0.02, f'{score:.3f}', ha='center')
    
    plt.title(f"{title}\n{song1} vs {song2}")
    plt.xlabel("N-gram Pair Position")
    plt.ylabel("Similarity Score")
    plt.ylim(0, 1.15)  # Make room for score labels
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add info box with consistent format
    info_text = (
        f'Total n-gram pairs: {len(similarities)}\n'
        f'Length of n-gram sequence: 7 (relative intervals)\n'
        f'Step Size: 4\n'
        f'Final Similarity Score: {avg_similarity:.4f}\n'
        f'Method: Sum Common'
    )
    plt.text(0.02, 0.98, info_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top',
             fontsize=9)
    
    plt.tight_layout()
    plt.show()

def plot_ngram_analysis(seq1, seq2, title, song1, song2, similarity_score):
    """Plot detailed n-gram analysis as a stacked bar chart with statistics."""
    # Convert sequences to sets of tuples for comparison
    tokens1 = {tuple(token) if isinstance(token, (list, np.ndarray)) else (token,) 
               for token in seq1 if token is not None}
    tokens2 = {tuple(token) if isinstance(token, (list, np.ndarray)) else (token,) 
               for token in seq2 if token is not None}
    
    # Calculate set operations
    common = len(tokens1.intersection(tokens2))
    unique_to_s1 = len(tokens1.difference(tokens2))
    unique_to_s2 = len(tokens2.difference(tokens1))
    
    # Create stacked bar chart
    plt.figure(figsize=(12, 6))
    
    # Create bars for both songs
    songs = ['Song 1', 'Song 2']
    common_values = [common, common]  # Common n-grams for both songs
    unique_values = [unique_to_s1, unique_to_s2]  # Unique n-grams for each song
    
    # Plot stacked bars
    bar_width = 0.35
    plt.bar(songs, common_values, bar_width, 
            label='Common N-grams', color='green', alpha=0.6)
    plt.bar(songs, unique_values, bar_width, 
            bottom=common_values, label='Unique N-grams', 
            color=['blue', 'red'], alpha=0.6)
    
    # Add value labels on bars
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
    
    # Customize plot
    plt.title(f"{title} N-gram Analysis\n{song1} vs {song2}\nSimilarity Score: {similarity_score}")
    plt.ylabel('Number of N-grams')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Add info box
    info_text = (
        f'Common N-grams: {common}\n'
        f'Unique to Song 1: {unique_to_s1}\n'
        f'Unique to Song 2: {unique_to_s2}\n'
        f'Total in Song 1: {common + unique_to_s1}\n'
        f'Total in Song 2: {common + unique_to_s2}\n'
        f'Similarity Score: {similarity_score}'
    )
    plt.text(0.02, 0.98, info_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top',
             fontsize=9)
    
    plt.tight_layout()
    plt.show()

def analyze_case(df, case_number):
    """Analyze a single case using Sum Common approach."""
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
        'Song 1': case_data.loc[0, 'File Name'],
        'Song 2': case_data.loc[1, 'File Name'],
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
    print(f"Pitch Similarity: {pitch_score}")
    print(f"Rhythm Similarity: {rhythm_score}")
    
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
        'Song 1': case_data.loc[0, 'File Name'],
        'Song 2': case_data.loc[1, 'File Name'],
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
    """Save similarity analysis results to CSV file."""
    df = pd.DataFrame(results)
    # Add binary ruling column
    df['Binary Ruling'] = (df['Ruling'] == 'Plagiarism').astype(int)
    # Extract case numbers and sort using raw string
    df['Case_Num'] = df['Case'].str.extract(r'(\d+)').astype(int)
    df = df.sort_values('Case_Num')
    df = df.drop('Case_Num', axis=1)
    # Reorder columns
    columns = ['Case', 'Ruling', 'Binary Ruling', 'Song 1', 'Song 2', 'Pitch Similarity', 'Rhythm Similarity']
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
                    print(f"Pitch Similarity: {result['Pitch Similarity']:.4f}")
                    print(f"Rhythm Similarity: {result['Rhythm Similarity']:.4f}")
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
    Key Differences from MelDet Approach:
    1. No cost matrix or edit distance calculation
    2. Calculates similarity per n-gram pair by counting common elements
    3. Calculates final similarity score as average of all n-gram pair scores
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
