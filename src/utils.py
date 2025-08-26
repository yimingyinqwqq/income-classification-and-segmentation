"""
Utility functions for income classification and segmentation project.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import warnings

warnings.filterwarnings("ignore")

# Set style for plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ["data", "models", "results"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def load_data(data_path, columns_path):
    """
    Load census data and column headers.

    Args:
        data_path (str): Path to the data file
        columns_path (str): Path to the columns file

    Returns:
        tuple: (dataframe, column_names)
    """
    try:
        # Load column names
        with open(columns_path, "r") as f:
            column_names = [line.strip() for line in f.readlines()]

        # Load data
        df = pd.read_csv(data_path, header=None, names=column_names)

        # Clean the target variable (last column)
        target_col = df.columns[-1]
        df[target_col] = df[target_col].str.strip()

        # Map target values to binary
        df[target_col] = df[target_col].map({"- 50000.": "<=50K", "50000+.": ">50K"})

        print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Target variable '{target_col}' distribution:")
        print(df[target_col].value_counts())
        print(
            f"Class imbalance ratio: {df[target_col].value_counts().iloc[0] / df[target_col].value_counts().iloc[1]:.1f}:1"
        )

        return df, column_names

    except FileNotFoundError as e:
        print(
            f"Error: Data files not found. Please ensure {data_path} and {columns_path} exist."
        )
        print("Creating sample data structure for demonstration...")
        return create_sample_data(), create_sample_columns()


def create_sample_data():
    """Create sample data for demonstration purposes."""
    np.random.seed(42)
    n_samples = 1000

    # Generate sample features (40 demographic variables)
    features = {}

    # Age (18-80)
    features["age"] = np.random.randint(18, 81, n_samples)

    # Work class
    work_classes = [
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked",
    ]
    features["workclass"] = np.random.choice(
        work_classes, n_samples, p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.03, 0.01, 0.01]
    )

    # Education level
    education_levels = [
        "Bachelors",
        "Some-college",
        "11th",
        "HS-grad",
        "Prof-school",
        "Assoc-acdm",
        "Assoc-voc",
        "9th",
        "7th-8th",
        "12th",
        "Masters",
        "1st-4th",
        "10th",
        "Doctorate",
        "5th-6th",
        "Preschool",
    ]
    features["education"] = np.random.choice(
        education_levels,
        n_samples,
        p=[
            0.2,
            0.25,
            0.1,
            0.25,
            0.05,
            0.05,
            0.05,
            0.02,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
        ],
    )

    # Marital status
    marital_status = [
        "Married-civ-spouse",
        "Divorced",
        "Never-married",
        "Separated",
        "Widowed",
        "Married-spouse-absent",
        "Married-AF-spouse",
    ]
    features["marital_status"] = np.random.choice(
        marital_status, n_samples, p=[0.45, 0.15, 0.25, 0.05, 0.05, 0.03, 0.02]
    )

    # Occupation
    occupations = [
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical",
        "Farming-fishing",
        "Transport-moving",
        "Priv-house-serv",
        "Protective-serv",
        "Armed-Forces",
    ]
    features["occupation"] = np.random.choice(
        occupations,
        n_samples,
        p=[
            0.1,
            0.15,
            0.15,
            0.15,
            0.1,
            0.1,
            0.05,
            0.05,
            0.05,
            0.03,
            0.03,
            0.02,
            0.02,
            0.01,
        ],
    )

    # Relationship
    relationships = [
        "Wife",
        "Own-child",
        "Husband",
        "Not-in-family",
        "Other-relative",
        "Unmarried",
    ]
    features["relationship"] = np.random.choice(
        relationships, n_samples, p=[0.2, 0.25, 0.2, 0.2, 0.1, 0.05]
    )

    # Race
    races = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
    features["race"] = np.random.choice(
        races, n_samples, p=[0.75, 0.1, 0.05, 0.05, 0.05]
    )

    # Sex
    features["sex"] = np.random.choice(["Male", "Female"], n_samples, p=[0.55, 0.45])

    # Hours per week (20-80)
    features["hours_per_week"] = np.random.randint(20, 81, n_samples)

    # Native country
    countries = [
        "United-States",
        "Mexico",
        "Philippines",
        "Germany",
        "Canada",
        "Puerto-Rico",
        "El-Salvador",
        "India",
        "Cuba",
        "England",
        "Jamaica",
        "South",
        "China",
        "Italy",
        "Dominican-Republic",
        "Vietnam",
        "Guatemala",
        "Japan",
        "Poland",
        "Columbia",
        "Taiwan",
        "Haiti",
        "Portugal",
        "Iran",
        "Nicaragua",
        "Peru",
        "Greece",
        "France",
        "Ecuador",
        "Ireland",
        "Hong",
        "Cambodia",
        "Trinadad&Tobago",
        "Laos",
        "Thailand",
        "Yugoslavia",
        "Outlying-US(Guam-USVI-etc)",
        "Hungary",
        "Honduras",
        "Scotland",
        "Holand-Netherlands",
    ]
    features["native_country"] = np.random.choice(
        countries, n_samples, p=[0.85] + [0.15 / 39] * 39
    )

    # Generate additional 30 demographic variables
    for i in range(30):
        var_name = f"demographic_var_{i+1}"
        # Mix of categorical and numerical variables
        if i % 3 == 0:  # Categorical
            features[var_name] = np.random.choice(
                ["A", "B", "C", "D"], n_samples, p=[0.4, 0.3, 0.2, 0.1]
            )
        else:  # Numerical
            features[var_name] = np.random.normal(0, 1, n_samples)

    # Create income labels based on features (simplified logic)
    income_labels = []
    for i in range(n_samples):
        # Simple rule: higher age, education, and hours increase income probability
        income_prob = (
            0.3 * (features["age"][i] - 18) / 62  # Age factor
            + 0.3 * (features["hours_per_week"][i] - 20) / 60  # Hours factor
            + 0.4 * np.random.random()  # Random factor
        )
        income_labels.append(">50K" if income_prob > 0.5 else "<=50K")

    features["income"] = income_labels

    # Add weight column (1.0 for all samples in demo)
    features["weight"] = np.ones(n_samples)

    df = pd.DataFrame(features)
    print(f"Sample data created: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def create_sample_columns():
    """Create sample column names."""
    columns = [
        "age",
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "hours_per_week",
        "native_country",
    ]

    # Add 30 demographic variables
    for i in range(30):
        columns.append(f"demographic_var_{i+1}")

    # Add target and weight
    columns.extend(["income", "weight"])

    return columns


def preprocess_data(df, target_col="income"):
    """
    Preprocess the data for machine learning.

    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of target column

    Returns:
        tuple: (X_processed, y, feature_names, preprocessor)
    """
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in [target_col, "weight"]]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    weights = df["weight"].copy() if "weight" in df.columns else None

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print(f"Categorical columns: {len(categorical_cols)}")
    print(f"Numerical columns: {len(numerical_cols)}")

    # Handle missing values
    X = X.fillna("Unknown" if categorical_cols else 0)

    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Scale numerical features
    scaler = StandardScaler()
    if numerical_cols:
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Encode target variable
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    print(f"Preprocessing complete. Final feature matrix shape: {X.shape}")

    return (
        X,
        y_encoded,
        X.columns.tolist(),
        {
            "label_encoders": label_encoders,
            "scaler": scaler,
            "target_encoder": target_encoder,
            "categorical_cols": categorical_cols,
            "numerical_cols": numerical_cols,
        },
    )


def evaluate_classification_model(y_true, y_pred, y_prob=None, model_name="Model"):
    """
    Evaluate classification model performance.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        model_name: Name of the model for reporting
    """
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS FOR {model_name.upper()}")
    print(f"{'='*50}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["<=50K", ">50K"]))

    # Calculate and print accuracy prominently
    accuracy = (y_true == y_pred).mean()
    print(f"\nACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)

    # ROC-AUC if probabilities are provided
    if y_prob is not None:
        roc_auc = roc_auc_score(y_true, y_prob)
        print(f"\nROC-AUC Score: {roc_auc:.4f}")

    return {
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True
        ),
        "confusion_matrix": cm,
        "roc_auc": roc_auc if y_prob is not None else None,
    }


def save_model(model, preprocessor, model_name, filepath):
    """
    Save trained model and preprocessor.

    Args:
        model: Trained model
        preprocessor: Preprocessing pipeline
        model_name: Name of the model
        filepath: Path to save the model
    """
    model_data = {
        "model": model,
        "preprocessor": preprocessor,
        "model_name": model_name,
    }
    joblib.dump(model_data, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load trained model and preprocessor.

    Args:
        filepath: Path to the saved model

    Returns:
        dict: Model data containing model, preprocessor, and model_name
    """
    model_data = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model_data


def plot_feature_importance(model, feature_names, model_name="Model", top_n=20):
    """
    Plot feature importance for tree-based models.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        model_name: Name of the model
        top_n: Number of top features to display
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(12, 8))
        plt.title(f"Top {top_n} Feature Importances - {model_name}")
        plt.bar(range(top_n), importances[indices])
        plt.xticks(
            range(top_n), [feature_names[i] for i in indices], rotation=45, ha="right"
        )
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.savefig(
            f'results/classification/feature_importance_{model_name.lower().replace(" ", "_")}.png',
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        print("Model does not have feature_importances_ attribute")


def analyze_segments(df, segment_labels, segment_name="Segment"):
    """
    Analyze characteristics of different segments.

    Args:
        df: Original dataframe
        segment_labels: Segment assignments
        segment_name: Name for the segmentation
    """
    df_with_segments = df.copy()
    df_with_segments["segment"] = segment_labels

    print(f"\n{'='*50}")
    print(f"SEGMENT ANALYSIS - {segment_name.upper()}")
    print(f"{'='*50}")

    # Segment sizes
    segment_sizes = df_with_segments["segment"].value_counts().sort_index()
    print(f"\nSegment Sizes:")
    for segment, size in segment_sizes.items():
        print(f"Segment {segment}: {size} ({size/len(df)*100:.1f}%)")

    # Analyze key variables by segment
    key_vars = ["age", "education", "occupation", "income", "hours_per_week"]
    available_vars = [var for var in key_vars if var in df.columns]

    for var in available_vars:
        print(f"\n{var.upper()} by Segment:")
        if df_with_segments[var].dtype in ["object", "category"]:
            # Categorical variable
            segment_analysis = (
                df_with_segments.groupby("segment")[var]
                .value_counts(normalize=True)
                .unstack(fill_value=0)
            )
            print(segment_analysis.round(3))
        else:
            # Numerical variable
            segment_analysis = df_with_segments.groupby("segment")[var].agg(
                ["mean", "std", "min", "max"]
            )
            print(segment_analysis.round(2))

    return df_with_segments


def create_segment_visualizations(df_with_segments, segment_name="Segment"):
    """
    Create visualizations for segment analysis.

    Args:
        df_with_segments: DataFrame with segment labels
        segment_name: Name for the segmentation
    """
    # Segment distribution
    plt.figure(figsize=(10, 6))
    segment_counts = df_with_segments["segment"].value_counts().sort_index()
    plt.bar(segment_counts.index, segment_counts.values)
    plt.title(f"Segment Distribution - {segment_name}")
    plt.xlabel("Segment")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        f'results/segmentation/segment_distribution_{segment_name.lower().replace(" ", "_")}.png',
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Income distribution by segment
    if "income" in df_with_segments.columns:
        plt.figure(figsize=(12, 6))
        income_segment = pd.crosstab(
            df_with_segments["segment"], df_with_segments["income"], normalize="index"
        )
        income_segment.plot(kind="bar", stacked=True)
        plt.title(f"Income Distribution by Segment - {segment_name}")
        plt.xlabel("Segment")
        plt.ylabel("Proportion")
        plt.legend(title="Income")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(
            f'results/segmentation/income_by_segment_{segment_name.lower().replace(" ", "_")}.png',
            dpi=300,
            bbox_inches="tight",
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
        plt.title(f"Age Distribution by Segment - {segment_name}")
        plt.xlabel("Age")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            f'results/segmentation/age_by_segment_{segment_name.lower().replace(" ", "_")}.png',
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
