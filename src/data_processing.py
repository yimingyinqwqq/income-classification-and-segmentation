"""
Data processing module for income classification and segmentation project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import (
    create_directories,
    load_data,
    preprocess_data,
    create_sample_data,
    create_sample_columns,
)
import warnings

warnings.filterwarnings("ignore")


def explore_data(df):
    """
    Perform comprehensive data exploration.

    Args:
        df (pd.DataFrame): Input dataframe
    """
    print("=" * 60)
    print("DATA EXPLORATION")
    print("=" * 60)

    # Basic information
    print(f"\nDataset Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Data types
    print(f"\nData Types:")
    print(df.dtypes.value_counts())

    # Missing values
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame(
        {"Missing Count": missing_values, "Missing Percent": missing_percent}
    ).sort_values("Missing Count", ascending=False)

    print(f"\nMissing Values:")
    print(missing_df[missing_df["Missing Count"] > 0])

    # Target variable analysis
    if "income" in df.columns:
        print(f"\nTarget Variable (Income) Distribution:")
        income_dist = df["income"].value_counts()
        print(income_dist)
        print(f"Class Imbalance: {income_dist.iloc[0] / income_dist.iloc[1]:.2f}:1")

        # Visualize target distribution
        plt.figure(figsize=(8, 6))
        income_dist.plot(kind="bar")
        plt.title("Income Distribution")
        plt.xlabel("Income Level")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(
            "results/classification/income_distribution.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Weight analysis
    if "weight" in df.columns:
        print(f"\nWeight Statistics:")
        print(df["weight"].describe())

        plt.figure(figsize=(10, 6))
        plt.hist(df["weight"], bins=50, alpha=0.7)
        plt.title("Distribution of Sample Weights")
        plt.xlabel("Weight")
        plt.ylabel("Frequency")
        plt.tight_layout()
        # plt.savefig("results/weight_distribution.png", dpi=300, bbox_inches="tight")
        # plt.close()

    # Key demographic variables analysis
    key_vars = [
        "age",
        "education",
        "occupation",
        "workclass",
        "marital_status",
        "race",
        "sex",
    ]
    available_vars = [var for var in key_vars if var in df.columns]

    print(f"\nKey Variables Analysis:")
    for var in available_vars:
        print(f"\n{var.upper()}:")
        if df[var].dtype in ["object", "category"]:
            # Categorical variable
            value_counts = df[var].value_counts()
            print(f"Unique values: {len(value_counts)}")
            print(f"Top 5 values:")
            print(value_counts.head())

            # Visualize if not too many categories
            if len(value_counts) <= 15:
                plt.figure(figsize=(12, 6))
                value_counts.plot(kind="bar")
                plt.title(f"{var.title()} Distribution")
                plt.xlabel(var.title())
                plt.ylabel("Count")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                # plt.savefig(
                #     f"results/{var}_distribution.png", dpi=300, bbox_inches="tight"
                # )
                # plt.close()
        else:
            # Numerical variable
            print(df[var].describe())

            plt.figure(figsize=(10, 6))
            plt.hist(df[var], bins=30, alpha=0.7)
            plt.title(f"{var.title()} Distribution")
            plt.xlabel(var.title())
            plt.ylabel("Frequency")
            plt.tight_layout()
            # plt.savefig(f"results/{var}_distribution.png", dpi=300, bbox_inches="tight")
            # plt.close()

    # Correlation analysis for numerical variables
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_cols) > 1:
        print(f"\nCorrelation Analysis:")
        correlation_matrix = df[numerical_cols].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            fmt=".2f",
        )
        plt.title("Correlation Matrix of Numerical Variables")
        plt.tight_layout()
        plt.savefig(
            "results/classification/correlation_matrix.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Income vs key variables analysis
    if "income" in df.columns:
        print(f"\nIncome vs Key Variables Analysis:")

        for var in available_vars[:5]:  # Limit to first 5 variables
            if df[var].dtype in ["object", "category"]:
                # Categorical variable
                income_cross = pd.crosstab(df[var], df["income"], normalize="index")
                print(f"\n{var.upper()} vs Income:")
                print(income_cross.round(3))

                # Visualize
                plt.figure(figsize=(12, 6))
                income_cross.plot(kind="bar", stacked=True)
                plt.title(f"Income Distribution by {var.title()}")
                plt.xlabel(var.title())
                plt.ylabel("Proportion")
                plt.legend(title="Income")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(
                    f"results/income_vs_{var}.png", dpi=300, bbox_inches="tight"
                )
                plt.close()
            else:
                # Numerical variable
                print(f"\n{var.upper()} vs Income:")
                income_stats = df.groupby("income")[var].describe()
                print(income_stats)

                # Visualize
                plt.figure(figsize=(10, 6))
                df.boxplot(column=var, by="income")
                plt.title(f"{var.title()} by Income Level")
                plt.suptitle("")  # Remove default suptitle
                plt.tight_layout()
                plt.savefig(
                    f"results/income_vs_{var}.png", dpi=300, bbox_inches="tight"
                )
                plt.close()


def analyze_data_quality(df):
    """
    Analyze data quality issues.

    Args:
        df (pd.DataFrame): Input dataframe
    """
    print("\n" + "=" * 60)
    print("DATA QUALITY ANALYSIS")
    print("=" * 60)

    # Duplicate analysis
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")

    # Outlier analysis for numerical variables
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_cols:
        print(f"\nOutlier Analysis (using IQR method):")
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")

    # Consistency checks
    print(f"\nData Consistency Checks:")

    # Age consistency
    if "age" in df.columns:
        invalid_age = df[df["age"] < 0]
        print(f"Invalid age values (< 0): {len(invalid_age)}")

    # Hours per week consistency
    if "hours_per_week" in df.columns:
        invalid_hours = df[df["hours_per_week"] < 0]
        print(f"Invalid hours per week (< 0): {len(invalid_hours)}")

    # Education vs age consistency
    if "education" in df.columns and "age" in df.columns:
        young_high_edu = df[
            (df["age"] < 18)
            & (df["education"].isin(["Bachelors", "Masters", "Doctorate"]))
        ]
        print(f"Young people with high education (< 18): {len(young_high_edu)}")


def prepare_data_for_modeling(df, target_col="label"):
    """
    Prepare data for machine learning modeling.

    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of target column

    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessor)
    """
    print("\n" + "=" * 60)
    print("DATA PREPARATION FOR MODELING")
    print("=" * 60)

    # For large datasets, we might want to sample for faster processing
    if len(df) > 50000:
        print(
            f"Large dataset detected ({len(df)} rows). Sampling for faster processing..."
        )
        # Sample 50K rows while maintaining class distribution
        df_sampled = (
            df.groupby(target_col)
            .apply(lambda x: x.sample(n=min(25000, len(x)), random_state=42))
            .reset_index(drop=True)
        )
        print(f"Sampled dataset: {len(df_sampled)} rows")
        df = df_sampled

    # Preprocess data
    X, y, feature_names, preprocessor = preprocess_data(df, target_col)

    # Split data
    from sklearn.model_selection import train_test_split

    # Use stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Training set class distribution: {np.bincount(y_train)}")
    print(f"Test set class distribution: {np.bincount(y_test)}")

    return X_train, X_test, y_train, y_test, preprocessor, feature_names


def main():
    """Main function to run data processing pipeline."""
    # Create directories
    create_directories()

    # Load data
    data_path = "data/census-bureau.data"
    columns_path = "data/census-bureau.columns"

    df, column_names = load_data(data_path, columns_path)

    # Explore data
    explore_data(df)

    # Analyze data quality
    analyze_data_quality(df)

    # Prepare data for modeling
    X_train, X_test, y_train, y_test, preprocessor, feature_names = (
        prepare_data_for_modeling(df, target_col="label")
    )

    # Save processed data
    import joblib

    processed_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "original_df": df,
    }
    joblib.dump(processed_data, "data/processed_data.pkl")
    print(f"\nProcessed data saved to data/processed_data.pkl")

    print(f"\nData processing complete!")
    print(f"Next steps:")
    print(f"1. Run classification_model.py to train income classification model")
    print(f"2. Run segmentation_model.py to create customer segments")


if __name__ == "__main__":
    main()
