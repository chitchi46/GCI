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
    print(f"\n--- 目的変数 '{target_col}' の分布 ---")
    counts = df[target_col].value_counts()
    percentages = df[target_col].value_counts(normalize=True) * 100
    
    dist_summary = pd.DataFrame({
        'カウント': counts,
        '割合 (%)': percentages
    })
    print(dist_summary)
    
    # 簡単な棒グラフで可視化 (Colab環境を想定し、表示されるようにする)
    try:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=target_col, data=df)
        plt.title(f"'{target_col}' の分布")
        plt.xlabel(target_col)
        plt.ylabel("カウント")
        # Colabで表示するために plt.show() を呼び出すか、main側で制御
        # ここでは一旦 main 側で plt.show() を呼ぶ想定で plt.show() は書かない
        # ただし、Colabで直接この関数をテストする場合は plt.show() が必要
        print(f"'{target_col}' の分布グラフの準備ができました。main.py側で表示してください。")
        plt.show()
    except Exception as e:
        print(f"グラフの描画中にエラーが発生しました: {e}")
        print("matplotlib や seaborn が正しくインストールされているか、GUI環境か確認してください。")

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