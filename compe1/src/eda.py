import pandas as pd
import matplotlib
matplotlib.use('Agg') # バックエンドを設定
import matplotlib.pyplot as plt
import seaborn as sns
import os # 追加

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
    # value_counts() はデフォルトでインデックスが値、値がカウントになる
    # これを DataFrame にする際は、reset_index() を使って整形することが多い
    dist_summary = df[target_col].value_counts().reset_index()
    dist_summary.columns = [target_col, 'カウント'] # 列名を適切に設定
    dist_summary['割合 (%)'] = (dist_summary['カウント'] / dist_summary['カウント'].sum()) * 100
    # target_col をインデックスにする場合（任意、表示のため）
    # dist_summary = dist_summary.set_index(target_col)
    print(dist_summary)
    
    try:
        plt.figure(figsize=(6, 4))
        # dist_summary を使ってプロットすると、集計済みのデータから作れる
        # sns.barplot(x=target_col, y='カウント', data=dist_summary.reset_index()) # reset_index() はset_indexした場合
        sns.countplot(x=target_col, data=df) # 元のデータでcountplotでもOK
        plt.title(f"'{target_col}' の分布")
        plt.xlabel(target_col)
        plt.ylabel("カウント")
        # plt.show() の代わりにファイルに保存
        image_path = "target_distribution.png"
        plt.savefig(image_path)
        plt.close() # メモリ解放のために図を閉じる
        print(f"'{target_col}' の分布グラフを {image_path} に保存しました。")
    except Exception as e:
        print(f"グラフの描画中にエラーが発生しました: {e}")
        print("matplotlib や seaborn が正しくインストールされているか確認してください。")

def visualize_feature_vs_target(df: pd.DataFrame, feature_col: str, target_col: str, output_dir: str = "results/eda_plots"):
    """特徴量と目的変数の関係を可視化し、ファイルに保存する関数"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    plt.figure(figsize=(10, 6))
    safe_feature_name = "".join(c if c.isalnum() else "_" for c in feature_col)
    save_path_base = os.path.join(output_dir, f"{safe_feature_name}_vs_{target_col}")

    print(f"Visualizing {feature_col} vs {target_col}...")

    if df[feature_col].dtype == 'object' or (pd.api.types.is_numeric_dtype(df[feature_col]) and df[feature_col].nunique() < 20):
        # カテゴリカル特徴量 (またはユニーク数が少ない数値特徴量)
        sns.countplot(x=feature_col, hue=target_col, data=df, palette="viridis")
        plt.title(f'{feature_col} vs {target_col} (Count Plot)', fontsize=15)
        plt.xticks(rotation=45, ha='right')
        current_save_path = f"{save_path_base}_countplot.png"
    elif pd.api.types.is_numeric_dtype(df[feature_col]):
        # 数値特徴量 - ヒストグラム
        sns.histplot(data=df, x=feature_col, hue=target_col, kde=True, multiple="stack", palette="viridis")
        plt.title(f'{feature_col} vs {target_col} (Histogram & KDE)', fontsize=15)
        current_save_path = f"{save_path_base}_histplot.png"
        plt.savefig(current_save_path)
        print(f"Saved plot: {current_save_path}")
        plt.close()

        # 数値特徴量 - ボックスプロット
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=target_col, y=feature_col, data=df, palette="viridis")
        plt.title(f'{feature_col} vs {target_col} (Box Plot)', fontsize=15)
        current_save_path = f"{save_path_base}_boxplot.png"
    else:
        print(f"Skipping visualization for {feature_col}: Not clearly categorical or numeric, or too many unique values.")
        plt.close()
        return

    plt.xlabel(feature_col, fontsize=12)
    plt.ylabel("Count" if "countplot" in current_save_path else feature_col, fontsize=12)
    plt.tight_layout()
    plt.savefig(current_save_path)
    print(f"Saved plot: {current_save_path}")
    plt.close()

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