#!/usr/bin/env python3
"""
Simple Customer Segmentation Model
Uses basic clustering on key demographic variables for marketing purposes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings

warnings.filterwarnings("ignore")


def create_directories():
    """Create necessary directories."""
    import os

    directories = ["data", "models", "results/segmentation"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def segmentation(df):
    """
    Customer segmentation using key demographic variables.

    Args:
        df: Original dataframe with demographic data

    Returns:
        tuple: (segments, segment_analysis)
    """
    print("=" * 60)
    print("CUSTOMER SEGMENTATION")
    print("=" * 60)

    # Select only key demographic variables for segmentation
    key_vars = ["age", "education", "occupation", "marital stat", "race", "sex"]
    available_vars = [var for var in key_vars if var in df.columns]

    print(f"Using {len(available_vars)} key variables: {available_vars}")

    # Create segmentation dataset
    seg_data = df[available_vars].copy()

    # Handle missing values
    seg_data = seg_data.fillna("Unknown")

    # Encode categorical variables
    categorical_cols = seg_data.select_dtypes(include=["object"]).columns
    numerical_cols = seg_data.select_dtypes(include=["int64", "float64"]).columns

    # Simple encoding for categorical variables
    for col in categorical_cols:
        le = LabelEncoder()
        seg_data[col] = le.fit_transform(seg_data[col].astype(str))

    # Scale numerical variables
    if len(numerical_cols) > 0:
        scaler = StandardScaler()
        seg_data[numerical_cols] = scaler.fit_transform(seg_data[numerical_cols])

    print(f"Segmentation data shape: {seg_data.shape}")

    # Simple K-Means clustering with 3 segments
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    segments = kmeans.fit_predict(seg_data)

    # Add segments to original dataframe
    df_with_segments = df.copy()
    df_with_segments["segment"] = segments

    print(f"\nSegmentation complete! Found {len(np.unique(segments))} segments")

    return segments, df_with_segments


def analyze_segments(df_with_segments):
    """
    Simple analysis of segment characteristics.

    Args:
        df_with_segments: DataFrame with segment labels

    Returns:
        dict: Segment analysis results
    """
    print("\n" + "=" * 60)
    print("SEGMENT ANALYSIS")
    print("=" * 60)

    # Basic segment statistics
    segment_sizes = df_with_segments["segment"].value_counts().sort_index()

    print("Segment Sizes:")
    for segment, size in segment_sizes.items():
        percentage = size / len(df_with_segments) * 100
        print(f"Segment {segment}: {size} people ({percentage:.1f}%)")

    # Analyze key variables by segment
    key_vars = ["age", "education", "occupation", "income"]
    available_vars = [var for var in key_vars if var in df_with_segments.columns]

    segment_analysis = {}

    for var in available_vars:
        print(f"\n{var.upper()} by Segment:")

        if df_with_segments[var].dtype in ["object", "category"]:
            # Categorical variable
            segment_stats = df_with_segments.groupby("segment")[var].agg(
                ["count", "nunique"]
            )
            print(segment_stats)

            # Most common value in each segment
            for segment in sorted(df_with_segments["segment"].unique()):
                segment_data = df_with_segments[df_with_segments["segment"] == segment]
                most_common = (
                    segment_data[var].mode().iloc[0]
                    if len(segment_data[var].mode()) > 0
                    else "Unknown"
                )
                print(f"  Segment {segment} most common {var}: {most_common}")
        else:
            # Numerical variable
            segment_stats = df_with_segments.groupby("segment")[var].agg(
                ["mean", "std", "min", "max"]
            )
            print(segment_stats.round(2))

        segment_analysis[var] = segment_stats

    return segment_analysis


def create_visualizations(df_with_segments):
    """
    Create visualizations for segments.

    Args:
        df_with_segments: DataFrame with segment labels
    """
    print("\nCreating visualizations...")

    # Segment distribution
    plt.figure(figsize=(10, 6))
    segment_counts = df_with_segments["segment"].value_counts().sort_index()
    plt.bar(segment_counts.index, segment_counts.values)
    plt.title("Customer Segment Distribution")
    plt.xlabel("Segment")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        "results/segmentation/segment_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Age distribution by segment
    if "age" in df_with_segments.columns:
        plt.figure(figsize=(12, 6))
        for segment in sorted(df_with_segments["segment"].unique()):
            segment_data = df_with_segments[df_with_segments["segment"] == segment][
                "age"
            ]
            plt.hist(segment_data, alpha=0.7, label=f"Segment {segment}", bins=20)
        plt.title("Age Distribution by Segment")
        plt.xlabel("Age")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            "results/segmentation/age_by_segment.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    # Income distribution by segment
    if "income" in df_with_segments.columns:
        plt.figure(figsize=(12, 6))
        income_segment = pd.crosstab(
            df_with_segments["segment"], df_with_segments["income"], normalize="index"
        )
        income_segment.plot(kind="bar", stacked=True)
        plt.title("Income Distribution by Segment")
        plt.xlabel("Segment")
        plt.ylabel("Proportion")
        plt.legend(title="Income")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(
            "results/segmentation/income_by_segment.png", dpi=300, bbox_inches="tight"
        )
        plt.close()


def generate_recommendations(df_with_segments):
    """
    Generate marketing recommendations based on segment characteristics.

    Args:
        df_with_segments: DataFrame with segment labels

    Returns:
        dict: Marketing recommendations for each segment
    """
    print("\n" + "=" * 60)
    print("MARKETING RECOMMENDATIONS")
    print("=" * 60)

    recommendations = {}

    for segment in sorted(df_with_segments["segment"].unique()):
        segment_data = df_with_segments[df_with_segments["segment"] == segment]

        print(f"\nSegment {segment} Marketing Strategy:")

        # Get segment characteristics
        avg_age = segment_data["age"].mean() if "age" in segment_data.columns else 0
        segment_size = len(segment_data)
        total_size = len(df_with_segments)
        segment_percentage = segment_size / total_size

        # Simple recommendations based on age and size
        recommendations[segment] = []

        # Age-based recommendations
        if avg_age < 35:
            recommendations[segment].append(
                "Target with digital marketing and social media"
            )
            recommendations[segment].append(
                "Offer student and young professional discounts"
            )
        elif avg_age < 50:
            recommendations[segment].append(
                "Focus on family-oriented products and services"
            )
            recommendations[segment].append("Emphasize value and convenience")
        else:
            recommendations[segment].append("Highlight quality and reliability")
            recommendations[segment].append(
                "Offer senior discounts and loyalty programs"
            )

        # Size-based recommendations
        if segment_percentage > 0.5:  # Majority segment
            recommendations[segment].append("Mass marketing campaigns")
            recommendations[segment].append("Focus on mainstream products")
        else:  # Minority segment
            recommendations[segment].append("Targeted niche marketing")
            recommendations[segment].append(
                "Specialized products and premium positioning"
            )

        # Print recommendations
        for i, rec in enumerate(recommendations[segment], 1):
            print(f"  {i}. {rec}")

    return recommendations


def main():
    """Main function to run segmentation."""
    # Create directories
    create_directories()

    # Load processed data
    try:
        processed_data = joblib.load("data/processed_data.pkl")
        original_df = processed_data["original_df"]
        print("Loaded processed data successfully.")

        # Sample data for faster processing
        if len(original_df) > 10000:
            print(
                f"Large dataset detected ({len(original_df)} rows). Sampling for segmentation..."
            )
            original_df = original_df.sample(n=10000, random_state=42)
            print(f"Sampled dataset: {len(original_df)} rows")

    except FileNotFoundError:
        print("Processed data not found. Please run data_processing.py first.")
        return

        # Run segmentation
    segments, df_with_segments = segmentation(original_df)

    # Analyze segments
    segment_analysis = analyze_segments(df_with_segments)

    # Create visualizations
    create_visualizations(df_with_segments)

    # Generate recommendations
    recommendations = generate_recommendations(df_with_segments)

    # Save results
    df_with_segments.to_csv("results/segmentation/segment_analysis.csv", index=False)

    # Save recommendations
    import json

    recommendations_serializable = {}
    for key, value in recommendations.items():
        recommendations_serializable[str(key)] = value
    with open("results/segmentation/marketing_recommendations.json", "w") as f:
        json.dump(recommendations_serializable, f, indent=2)

    print(f"\nSegmentation complete!")
    print(f"Results saved to results/segmentation/ directory")


if __name__ == "__main__":
    main()
