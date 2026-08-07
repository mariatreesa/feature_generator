"""CLI entry point for running the feature extraction pipeline."""

import argparse
from abcam_task.generate_pipeline import FeatureGeneratePipeline

def main():
    """
    Takes input path and output path from command line and save
    the new features as parquet to output path
    
    """
    parser = argparse.ArgumentParser(description="Run sequence feature extraction pipeline.")
    parser.add_argument("--input", required=True, help="Path to uniprot_sequences.csv")
    parser.add_argument("--output", default="new_features.parquet", help="Path to output feature file")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}...")
    pipeline = FeatureGeneratePipeline()
    features_df = pipeline.run_pipeline(args.input)
    
    # Save output Parquet for high performance retrieval
    features_df.to_parquet(args.output, index=False)

if __name__ == "__main__":
    main()