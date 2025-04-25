import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

def load_squared_errors():
    """Load individual squared errors from CSV."""
    base_path = Path(__file__).parent
    try:
        df = pd.read_csv(base_path / 'individual_squared_errors.csv')
        return df
    except Exception as e:
        print(f"Error loading individual squared errors: {e}")
        raise

def perform_kendall_w_analysis(squared_errors_df, feature='Pitch'):
    """Perform Kendall's W analysis using individual squared errors."""
    # Pivot the data to get approaches as columns and cases as rows
    feature_key = 'Pitch_Squared_Error' if feature == 'Pitch' else 'Rhythm_Squared_Error'
    pivot_df = squared_errors_df.pivot(index='Case', columns='Approach', values=feature_key)
    
    # Print actual squared errors for inspection
    print(f"\nSquared Errors for {feature}:")
    print(pivot_df.round(4))
    
    # Create and print rankings
    rankings = pivot_df.rank(axis=1)
    print(f"\nRankings for {feature}:")
    print(rankings.round(2))
    
    # Calculate rank variance per case
    rank_variance_per_case = rankings.var(axis=1)
    print(f"\nRank variance per case:")
    print(rank_variance_per_case.round(4))
    print(f"Average rank variance: {rank_variance_per_case.mean():.4f}")
    
    n = len(rankings)  # number of cases
    k = len(rankings.columns)  # number of approaches
    
    # Calculate mean rankings
    mean_rankings = rankings.mean()
    
    # Calculate Kendall's W
    rank_sums = rankings.sum()
    s = np.var(rank_sums) * (k-1)
    w = (12 * s) / (k * k * (n * n * n - n))
    
    # Calculate chi-square statistic for p-value
    chi2 = k * (n-1) * w
    df = k - 1
    p_value = 1 - stats.chi2.cdf(chi2, df)
    
    print(f"\nKendall's W Analysis Results for {feature} Squared Errors:")
    print("-" * 50)
    print(f"Number of cases: {n}")
    print(f"Number of approaches: {k}")
    print(f"Kendall's W: {w:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    print("\nMean Rankings per Approach:")
    for approach, rank in mean_rankings.items():
        print(f"{approach}: {rank:.2f}")
    
    return w, p_value, mean_rankings.to_dict()

def save_kendalls_w_results(squared_errors_df):
    """Save Kendall's W analysis results to CSV."""
    results = []
    
    # Perform analysis for both Pitch and Rhythm
    for feature in ['Pitch', 'Rhythm']:
        w, p_value, rankings = perform_kendall_w_analysis(squared_errors_df, feature)
        
        for approach, rank in rankings.items():
            results.append({
                'Feature': feature,
                'Approach': approach,
                'Mean_Rank': f"{rank:.2f}",
                'Kendalls_W': f"{w:.4f}",
                'P_Value': f"{p_value:.4f}",
                'Significant': 'Yes' if p_value < 0.05 else 'No'
            })
    
    # Save to CSV
    kendalls_path = Path(__file__).parent / "kendalls_w_results.csv"
    pd.DataFrame(results).to_csv(kendalls_path, index=False)
    print(f"\nKendall's W analysis results saved to: {kendalls_path}")

def main():
    """Main execution for Kendall's W analysis."""
    print("\nLoading individual squared errors...")
    squared_errors_df = load_squared_errors()
    
    # Perform Kendall's W analysis and save results
    save_kendalls_w_results(squared_errors_df)

if __name__ == "__main__":
    main()