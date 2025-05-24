import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ast
from collections import Counter

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

def calculate_cosine_similarity(seq1, seq2):
    """
    Calculate cosine similarity between two sequences.
    Cosine similarity measures the cosine of the angle between two vectors, providing a similarity score between 0 and 1.
    
    Process Overview:
    1. Convert sequences to frequency vectors
    2. Calculate dot product of vectors
    3. Calculate magnitudes of vectors
    4. Compute cosine similarity: dot_product / (magnitude1 * magnitude2)
    
    Score Interpretation:
    - 1.0: Sequences are identical (0° angle)
    - 0.0: Sequences are completely different (90° angle)
    - Values between 0-1: Partial similarity
    """
    try:
        # Step 1: Input validation
        # Ensure inputs are valid lists or numpy arrays
        if not isinstance(seq1, (list, np.ndarray)) or not isinstance(seq2, (list, np.ndarray)):
            return 0.0
            
        # Step 2: Convert sequences to tokens
        # Handle both numeric and string elements by hashing strings
        tokens1 = [token if isinstance(token, (int, float)) else hash(str(token)) 
                  for token in seq1 if token is not None]
        tokens2 = [token if isinstance(token, (int, float)) else hash(str(token)) 
                  for token in seq2 if token is not None]
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Step 3: Create frequency vectors
        # Convert sequences to vectors where each element represents frequency of occurrence
        unique_elements = list(set(tokens1 + tokens2))  # Get all unique elements
        vec1 = np.array([tokens1.count(e) for e in unique_elements])  # Frequency vector for seq1
        vec2 = np.array([tokens2.count(e) for e in unique_elements])  # Frequency vector for seq2
        
        # Step 4: Calculate cosine similarity components
        # Dot product: sum of element-wise multiplication of vectors
        dot_product = np.dot(vec1, vec2)
        
        # Calculate vector magnitudes (Euclidean norm)
        norm1 = np.linalg.norm(vec1)  # sqrt(sum of squared elements)
        norm2 = np.linalg.norm(vec2)
        
        # Step 5: Handle zero division case
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        # Step 6: Compute final similarity score
        # cos(θ) = dot_product / (||vec1|| * ||vec2||)
        similarity = dot_product / (norm1 * norm2)
        return round(similarity, 4)  # Round to 4 decimal places
        
    except:
        return 0.0  # Return 0 for any unexpected errors

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
    """Plot similarity value with formula explanation."""
    plt.figure(figsize=(12, 6))
    
    # Convert to percentage for display
    similarities_pct = similarities * 100
    
    # Create single bar for similarity score
    plt.bar(['Cosine Similarity'], [similarities_pct], color='skyblue', alpha=0.6)
    
    # Add score label
    plt.text(0, similarities_pct + 2, f'{similarities_pct:.2f}%', ha='center')
    
    plt.title(f"Cosine Similarity Score for \n{title}\n{song1} vs {song2}")
    plt.ylabel("Similarity Score (%)")
    plt.ylim(0, 115)
    plt.grid(True, alpha=0.3)
    
    # Add formula explanation with percentage
    info_text = (
        f'Cosine Similarity Score: {similarities_pct:.2f}% ({interpret_similarity(similarities_pct)})\n'
        f'Method: Cosine Similarity\n'
    )
    
    plt.text(1.05, 0.98, info_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top',
             fontsize=9)
    
    plt.subplots_adjust(right=0.75)
    plt.show()

def plot_ngram_analysis(seq1, seq2, title, song1, song2, similarity_score):
    """Plot detailed n-gram frequency analysis."""
    plt.figure(figsize=(15, 8))
    
    # Convert sequences to frequency vectors
    tokens1 = [token if isinstance(token, (int, float)) else hash(str(token)) 
               for token in seq1 if token is not None]
    tokens2 = [token if isinstance(token, (int, float)) else hash(str(token)) 
               for token in seq2 if token is not None]
    
    # Calculate vectors for display
    unique_elements = sorted(list(set(tokens1 + tokens2)))
    vec1 = np.array([tokens1.count(e) for e in unique_elements])
    vec2 = np.array([tokens2.count(e) for e in unique_elements])
    
    # Get unique elements and their frequencies
    unique_elements = list(set(tokens1 + tokens2))
    freq1 = Counter(tokens1)
    freq2 = Counter(tokens2)
    
    # Calculate frequencies for visualization
    common_elements = set(tokens1) & set(tokens2)
    unique_to_s1 = set(tokens1) - set(tokens2)
    unique_to_s2 = set(tokens2) - set(tokens1)
    
    # Create visualization
    songs = ['Song A', 'Song B']
    common = len(common_elements)
    unique_s1 = len(unique_to_s1)
    unique_s2 = len(unique_to_s2)
    
    common_values = [common, common]
    unique_values = [unique_s1, unique_s2]
    
    bar_width = 0.35
    plt.bar(songs, common_values, bar_width, 
            label='Common Elements', color='green', alpha=0.6)
    plt.bar(songs, unique_values, bar_width, 
            bottom=common_values, label='Unique Elements', 
            color=['blue', 'red'], alpha=0.6)
    
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
    
    # Format score for display
    similarity_pct = similarity_score * 100
    
    plt.title(f"{title} Feature Analysis\n{song1} vs {song2}\n"
              f"Cosine Similarity Score: {similarity_pct:.2f}% ({interpret_similarity(similarity_pct)})")
    plt.ylabel('Number of Elements')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add detailed calculation box with example
    calc_text = (
        f'Cosine Similarity Analysis:\n\n'
        f'1. Vector Analysis:\n'
        f'   - Song A: {len(tokens1)} elements\n'
        f'   - Song B: {len(tokens2)} elements\n\n'
        f'2. Element Distribution:\n'
        f'   Common elements: {common}\n'
        f'   Unique to Song A: {unique_s1}\n'
        f'   Unique to Song B: {unique_s2}\n\n'
        f'3. Cosine Similarity Score:\n'
        f'   cos(θ) = A·B / (||A|| ||B||)\n'
        f'   Example calculation:\n'
        f'   A = {vec1[:3]}... (frequency vector)\n'
        f'   B = {vec2[:3]}... (frequency vector)\n'
        f'   A·B = {np.dot(vec1, vec2):.2f} (dot product)\n'
        f'   ||A|| = {np.linalg.norm(vec1):.2f} (magnitude)\n'
        f'   ||B|| = {np.linalg.norm(vec2):.2f} (magnitude)\n'
        f'   Score = {similarity_score * 100:.2f}%'
    )
    
    plt.text(1.15, 0.98, calc_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top',
             fontsize=9)
    
    plt.subplots_adjust(right=0.7)
    plt.show()

def analyze_case(df, case_number):
    """Analyze a single case using original Tversky measure."""
    case_data = df[df['Case'] == case_number].reset_index(drop=True)
    
    if len(case_data) != 2:
        print(f"Error: Case {case_number} does not have exactly 2 songs")
        return None
    
    # Get sequences
    seq1_pitch = case_data.loc[0, 'Relative Pitch']
    seq2_pitch = case_data.loc[1, 'Relative Pitch']
    seq1_rhythm = case_data.loc[0, 'Relative Rhythm']
    seq2_rhythm = case_data.loc[1, 'Relative Rhythm']
    
    # Calculate similarity scores
    pitch_score = calculate_cosine_similarity(seq1_pitch, seq2_pitch)
    rhythm_score = calculate_cosine_similarity(seq1_rhythm, seq2_rhythm)
    
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
    
    # Calculate similarity scores
    pitch_score = calculate_cosine_similarity(seq1_pitch, seq2_pitch)
    rhythm_score = calculate_cosine_similarity(seq1_rhythm, seq2_rhythm)
    
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
    """Analyze all cases using original Tversky measure."""
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
    """Interactive menu for Tversky analysis."""
    while True:
        print("\nTversky Analysis Options:")
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
    """Main execution for cosine similarity measure."""
    # Load sequences from library
    df = load_sequences_from_library()
    
    # Analyze all cases
    print("Analyzing all cases using cosine similarity measure...")
    results = analyze_all_cases(df)
    
    # Save results to CSV
    report_path = Path(__file__).parent / "similarity_report_cosine.csv"
    save_similarity_report(results, report_path)
    
    print(f"\nAnalysis completed. Results saved to {report_path}")
    
    # Start interactive menu
    interactive_menu(df)

if __name__ == "__main__":
    main()