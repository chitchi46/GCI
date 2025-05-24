import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataframe(df: pd.DataFrame, df_name: str):
    print(f"--- Analyzing DataFrame: {df_name} ---")
    
    print("\n### Shape:")
    print(df.shape)
    
    print("\n### First 5 rows:")
    print(df.head())
    
    print("\n### Data Types:")
    print(df.dtypes)
    
    print("\n### Basic Statistics (Numerical Features):")
    print(df.describe())
    
    print("\n### Missing Value Counts and Percentages:")
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_info = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
    print(missing_info[missing_info['Missing Values'] > 0].sort_values(by='Percentage', ascending=False))
    
    if 'Perished' in df.columns:
        print("\n### Target Variable (Perished) Distribution:")
        target_distribution = df['Perished'].value_counts()
        target_percentage = df['Perished'].value_counts(normalize=True) * 100
        target_info = pd.DataFrame({'Count': target_distribution, 'Percentage': target_percentage})
        print(target_info)
        
    print("\n### Categorical Feature Value Counts (Illustrative - top 5):")
    for col in df.select_dtypes(include=['object', 'category']).columns:
        print(f"\n--- Feature: {col} ---")
        print(df[col].value_counts(dropna=False).head())

    print(f"--- End of Analysis for {df_name} ---\n")

def summarize_target_distribution(df: pd.DataFrame, target_col: str):
    """目的変数の分布を集計・表示する関数"""
    # TODO: 実装
    pass

def visualize_feature_vs_target(df: pd.DataFrame, feature_col: str, target_col: str):
    """特徴量と目的変数の関係を可視化する関数"""
    # TODO: 実装
    pass

if __name__ == '__main__':
    print("Starting EDA script...")
    
    # Load data
    try:
        df_train = pd.read_csv("../data/train.csv")
        print("train.csv loaded successfully.")
    except FileNotFoundError:
        print("Error: train.csv not found. Make sure it's in the 'compe1/data/' directory.")
        exit()
        
    try:
        df_test = pd.read_csv("../data/test.csv")
        print("test.csv loaded successfully.")
    except FileNotFoundError:
        print("Error: test.csv not found. Make sure it's in the 'compe1/data/' directory.")
        exit()

    analyze_dataframe(df_train, "Training Data")
    analyze_dataframe(df_test, "Test Data")

    print("\nEDA script finished.") 