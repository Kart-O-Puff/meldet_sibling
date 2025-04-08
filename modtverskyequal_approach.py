import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ast
from collections import Counter

def load_sequences_from_library():
    """Load and parse the melody library CSV file."""
    csv_path = Path(__file__).parent / "MCIC_Dataset" / "MCIC_Preprocessed" / "melody_library.csv"
    df = pd.read_csv(csv_path)
    
    # Convert string representations back to actual lists
    df['Relative Pitch'] = df['Relative Pitch'].apply(ast.literal_eval)  # Changed back to Relative Pitch
    df['Relative Rhythm'] = df['Relative Rhythm'].apply(ast.literal_eval)  # Changed back to Relative Rhythm
    
    # Ensure 'Ruling' column exists
    if 'Category' in df.columns and 'Ruling' not in df.columns:
        df['Ruling'] = df['Category']
        df = df.drop('Category', axis=1)
    
    return df

def calculate_modified_tversky(seq1, seq2, alpha=None, beta=None):
    """
    Calculate Modified Tversky-Equal similarity between tokenized sequences.
    Each sequence is treated as a document containing n-gram tokens.
    """
    try:
        # Convert n-grams to tuples for hashable comparison
        tokens1 = [tuple(token) if isinstance(token, (list, np.ndarray)) else (token,) 
                  for token in seq1 if token is not None]
        tokens2 = [tuple(token) if isinstance(token, (list, np.ndarray)) else (token,) 
                  for token in seq2 if token is not None]
        
        # Validate input silently
        if not tokens1 or not tokens2:
            return 0.0
        
        # Count token frequencies
        counter1 = Counter(tokens1)
        counter2 = Counter(tokens2)
        
        # Find common tokens
        common_tokens = set(counter1.keys()) & set(counter2.keys())
        if not common_tokens:
            return 0.0
        
        # Calculate sequence lengths
        len1, len2 = len(tokens1), len(tokens2)
        
        # Compute weights if not provided
        if alpha is None or beta is None:
            alpha = len1 / (len1 + len2)
            beta = len2 / (len1 + len2)
        
        # Calculate salience scores for common tokens
        salience_scores = []
        for token in common_tokens:
            try:
                # Calculate normalized frequencies
                p_A = counter1[token] / len1
                p_B = counter2[token] / len2
                
                # Calculate salience score
                denominator = p_A + (alpha * (1 - p_A)) + (beta * (1 - p_B))
                if denominator == 0:
                    continue
                    
                salience = p_A / denominator
                salience_scores.append(salience)
            except:
                continue
        
        if not salience_scores:
            return 0.0
        
        # Calculate final similarity score
        similarity = sum(salience_scores) / len(salience_scores)
        return round(similarity, 4)
        
    except:
        return 0.0

def plot_similarity_comparison(similarities, title, song1, song2):
    """Plot bar chart showing similarity scores for each n-gram pair."""
    plt.figure(figsize=(12, 6))
    
    # Create bar plot for n-gram similarities
    x = range(len(similarities))
    plt.bar(x, similarities, color='skyblue', alpha=0.6, 
            label='Modified Tversky-Equal Similarity')
    
    # Add average line
    avg_similarity = np.mean(similarities)
    plt.axhline(y=avg_similarity, color='red', linestyle='--', 
                label=f'Average Similarity: {avg_similarity:.4f}')
    
    # Add individual score labels
    for i, score in enumerate(similarities):
        plt.text(i, score + 0.02, f'{score:.3f}', ha='center')
    
    plt.title(f"{title}\n{song1} vs {song2}")
    plt.xlabel("N-gram Pair Position")
    plt.ylabel("Similarity Score")
    plt.ylim(0, 1.15)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add info box
    info_text = (
        f'Total n-gram pairs: {len(similarities)}\n'
        f'Length of n-gram sequence: 7 (relative intervals)\n'
        f'Step Size: 4\n'
        f'Final Similarity Score: {avg_similarity:.4f}\n'
        f'Method: Modified Tversky-Equal (α=β=0.5)'
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
        # Label for common portion
        plt.text(i, common_values[i]/2, 
                f'Common: {common_values[i]}', 
                ha='center', va='center')
        # Label for unique portion
        plt.text(i, common_values[i] + unique_values[i]/2,
                f'Unique: {unique_values[i]}',
                ha='center', va='center')
        # Total label at top
        plt.text(i, common_values[i] + unique_values[i],
                f'Total: {common_values[i] + unique_values[i]}',
                ha='center', va='bottom')
    
    # Customize plot
    plt.title(f"{title} N-gram Analysis\n{song1} vs {song2}\nSimilarity Score: {similarity_score * 100:.2f}%")
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
    """Analyze a single case using Modified Tversky-Equal approach."""
    case_data = df[df['Case'] == case_number].reset_index(drop=True)
    
    if len(case_data) != 2:
        print(f"Error: Case {case_number} does not have exactly 2 songs")
        return None
    
    # Get sequences (entire documents)
    doc1_pitch = case_data.loc[0, 'Relative Pitch']
    doc2_pitch = case_data.loc[1, 'Relative Pitch']
    doc1_rhythm = case_data.loc[0, 'Relative Rhythm']
    doc2_rhythm = case_data.loc[1, 'Relative Rhythm']
    
    # Calculate single document-level similarity score
    pitch_score = calculate_modified_tversky(doc1_pitch, doc2_pitch)
    rhythm_score = calculate_modified_tversky(doc1_rhythm, doc2_rhythm)
    
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
    
    # Get sequences (changed to relative intervals)
    seq1_pitch = case_data.loc[0, 'Relative Pitch']
    seq2_pitch = case_data.loc[1, 'Relative Pitch']
    seq1_rhythm = case_data.loc[0, 'Relative Rhythm']
    seq2_rhythm = case_data.loc[1, 'Relative Rhythm']
    
    # Calculate single document-level similarity score without rounding
    pitch_score = calculate_modified_tversky(seq1_pitch, seq2_pitch)
    rhythm_score = calculate_modified_tversky(seq1_rhythm, seq2_rhythm)
    
    print(f"\nFinal Similarity Scores for {case_number}:")
    print(f"Pitch Similarity: {pitch_score * 100:.2f}%")
    print(f"Rhythm Similarity: {rhythm_score * 100:.2f}%")
    
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
    """Analyze all cases using Modified Tversky-Equal approach."""
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
    """Interactive menu for Modified Tversky-Equal analysis."""
    while True:
        print("\nModified Tversky-Equal Analysis Options:")
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
    """Main execution for Modified Tversky-Equal approach."""
    # Load sequences from library
    df = load_sequences_from_library()
    
    # Analyze all cases without printing updates
    print("Analyzing all cases using Modified Tversky-Equal approach...")
    results = analyze_all_cases(df)
    
    # Save results to CSV
    report_path = Path(__file__).parent / "similarity_report_modtverskyequal.csv"
    save_similarity_report(results, report_path)
    
    print(f"\nAnalysis completed. Results saved to {report_path}")
    
    # Start interactive menu
    interactive_menu(df)

if __name__ == "__main__":
    main()
