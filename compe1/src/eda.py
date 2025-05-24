import pandas as pd
import matplotlib
matplotlib.use('Agg') # バックエンドを設定
import matplotlib.pyplot as plt
import seaborn as sns
import os # 追加
import numpy as np

from src import config # EDAプロットのディレクトリ等を参照する可能性を考慮

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
    """目的変数の分布を集計し表示する"""
    print(f"\n--- Target Variable Distribution ({target_col}) ---")
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in DataFrame.")
        return
    
    value_counts = df[target_col].value_counts()
    percentage = df[target_col].value_counts(normalize=True) * 100
    
    dist_summary = pd.DataFrame({
        'Count': value_counts,
        'Percentage': percentage.round(2)
    })
    print(dist_summary)
    print(f"Total observations: {len(df)}")

    # 簡単なテキストベースの棒グラフもどき
    try:
        max_count_len = len(str(value_counts.max()))
        max_label_len = len(str(value_counts.index.max())) # 数値ラベルの場合を考慮
        if isinstance(value_counts.index, pd.CategoricalIndex) or df[target_col].dtype == 'object':
             max_label_len = value_counts.index.astype(str).map(len).max()

        print("\nVisual representation:")
        for label, count in value_counts.items():
            bar = '#' * int((count / value_counts.max()) * 30) # 30文字幅で正規化
            print(f"{str(label).ljust(max_label_len)} | {str(count).rjust(max_count_len)} | {bar}")
    except Exception as e:
        print(f"(Could not generate text bar graph: {e})")

def visualize_feature_vs_target(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    output_dir: str = "results/eda_plots", # config.EDA_PLOTS_DIR をデフォルトにすることも検討
    filename_prefix: str = "" 
):
    """
    指定された特徴量と目的変数の関係を可視化し、ファイルに保存する。
    特徴量の型に応じて、カウントプロット（カテゴリカル）またはヒストグラム/ボックスプロット（数値）を生成する。
    """
    if feature_col not in df.columns:
        print(f"Warning: Feature column '{feature_col}' not found in DataFrame. Skipping visualization.")
        return
    if target_col not in df.columns:
        print(f"Warning: Target column '{target_col}' not found in DataFrame. Skipping visualization.")
        return

    # 出力ディレクトリが存在しない場合は作成
    # この処理は呼び出し側 (e.g., main.py) で一度だけ行う方が効率的かもしれない
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created directory for plots: {output_dir}")
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}. Plots will not be saved.")
            # 保存先がない場合はプロット表示だけ試みるか、ここで終了するか。
            # GUI環境がないColab等では表示も難しいので、保存できないなら終了が妥当か。
            # ここでは続行し、savefigでエラーになるのに任せる。
            pass # または return

    plt.style.use('seaborn-v0_8-darkgrid') # スタイルの指定
    # 日本語フォント設定 (matplotlib と seaborn で必要に応じて)
    # plt.rcParams['font.family'] = 'IPAexGothic' # 例: IPAexGothic がインストールされている場合
    # sns.set(font='IPAexGothic') # seaborn用

    print(f"Visualizing {feature_col} vs {target_col} (prefix: '{filename_prefix}')...")

    # ファイル名に使えない文字を置換するためのヘルパー
    def sanitize_filename(name_part):
        return "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in str(name_part))

    safe_feature_name = sanitize_filename(feature_col)
    safe_target_name = sanitize_filename(target_col)

    try:
        # 特徴量の型に基づいてプロットタイプを決定
        # nunique() が非常に多い場合も数値型として扱う方が適切な場合がある
        if df[feature_col].dtype == 'object' or df[feature_col].nunique() < 20:  # カテゴリカルと判断
            fig, ax = plt.subplots(figsize=(12, 7)) # サイズ調整
            # データ型や欠損値に注意
            order = sorted(df[feature_col].dropna().unique()) if df[feature_col].nunique() < 20 else None

            sns.countplot(x=feature_col, hue=target_col, data=df, ax=ax, palette="viridis", order=order)
            
            ax.set_title(f'{feature_col} vs {target_col} (Count Plot)', fontsize=16)
            ax.set_xlabel(feature_col, fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.legend(title=target_col)
            ax.tick_params(axis='x', rotation=45, ha='right') # ラベルが重ならないように調整
            plt.tight_layout() # レイアウト調整
            
            save_path_bar = os.path.join(output_dir, f"{filename_prefix}{safe_feature_name}_vs_{safe_target_name}_barplot.png")
            plt.savefig(save_path_bar)
            plt.close(fig) # メモリ解放
            print(f"Bar plot saved to {save_path_bar}")

        else:  # 数値と判断 (nunique() >= 20 or 数値型)
            # 1. ヒストグラム (目的変数別)
            fig_hist, ax_hist = plt.subplots(figsize=(12, 7))
            try: # KDE計算でエラーが出ることがある (e.g. データ点が少なすぎる)
                sns.histplot(data=df, x=feature_col, hue=target_col, kde=True, multiple="stack", ax=ax_hist, palette="viridis", bins=30) # bins調整
            except RuntimeError as e_kde:
                print(f"Could not plot KDE for {feature_col} (Error: {e_kde}). Plotting without KDE.")
                sns.histplot(data=df, x=feature_col, hue=target_col, kde=False, multiple="stack", ax=ax_hist, palette="viridis", bins=30)

            ax_hist.set_title(f'{feature_col} Distribution by {target_col}', fontsize=16)
            ax_hist.set_xlabel(feature_col, fontsize=12)
            ax_hist.set_ylabel('Frequency', fontsize=12)
            ax_hist.legend(title=target_col)
            plt.tight_layout()
            
            save_path_hist = os.path.join(output_dir, f"{filename_prefix}{safe_feature_name}_distribution.png")
            fig_hist.savefig(save_path_hist)
            plt.close(fig_hist)
            print(f"Histogram saved to {save_path_hist}")

            # 2. ボックスプロット (目的変数別)
            fig_box, ax_box = plt.subplots(figsize=(10, 7)) # 少し幅を狭く
            sns.boxplot(x=target_col, y=feature_col, data=df, ax=ax_box, palette="viridis", showfliers=True) # 外れ値も表示
            ax_box.set_title(f'{feature_col} vs {target_col} (Box Plot)', fontsize=16)
            ax_box.set_xlabel(target_col, fontsize=12) # X軸はtarget_col
            ax_box.set_ylabel(feature_col, fontsize=12) # Y軸はfeature_col
            # X軸のラベルが数値の場合、intに変換して見栄えを良くする (例: 0.0 -> 0)
            # ただし、target_col が必ずしも数値とは限らないため、型チェックを入れるか、
            # 呼び出し側で適切に処理されたものが渡される前提とする。ここでは汎用的にそのまま表示。
            # current_xticklabels = [item.get_text() for item in ax_box.get_xticklabels()]
            # try:
            #    ax_box.set_xticklabels([int(float(label)) for label in current_xticklabels])
            # except ValueError:
            #    pass # 変換できない場合はそのまま
            
            plt.tight_layout()

            save_path_box = os.path.join(output_dir, f"{filename_prefix}{safe_feature_name}_vs_{safe_target_name}_boxplot.png")
            fig_box.savefig(save_path_box)
            plt.close(fig_box)
            print(f"Box plot saved to {save_path_box}")
            
    except Exception as e:
        print(f"An error occurred during visualization of {feature_col}: {e}")
        # エラーが発生した場合、作成途中の図が残らないように閉じる
        if 'fig' in locals() and fig is not None: plt.close(fig)
        if 'fig_hist' in locals() and fig_hist is not None: plt.close(fig_hist)
        if 'fig_box' in locals() and fig_box is not None: plt.close(fig_box)

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

    # 以下、ユーザーがColabで実行するためのサンプルコード (このファイルが直接実行された場合)
    print("Running eda.py standalone for testing purposes...")

    # ダミーデータフレームの作成
    data_size = 200
    test_data = pd.DataFrame({
        'Age': np.random.randint(1, 80, data_size),
        'Fare': np.random.lognormal(3, 1, data_size) * 10,
        'Pclass': np.random.choice([1, 2, 3], data_size, p=[0.3, 0.3, 0.4]),
        'Sex': np.random.choice(['male', 'female'], data_size),
        'Embarked': np.random.choice(['S', 'C', 'Q', 'S', 'S'], data_size), # Sを多めに
        'Perished': np.random.choice([0, 1], data_size)
    })
    test_data['Cabin'] = np.random.choice(['A12', 'B45', 'C78', 'M', 'M', 'D20', 'M'], data_size) # Mは欠損扱い
    test_data.loc[test_data.sample(frac=0.1).index, 'Age'] = np.nan # Ageに欠損を導入
    test_data.loc[test_data.sample(frac=0.05).index, 'Embarked'] = np.nan # Embarkedに欠損を導入

    # 欠損値の簡単な補完 (テスト用)
    test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
    test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace=True)
    
    # テスト用の出力ディレクトリ
    TEST_OUTPUT_DIR = "results/eda_plots_test"
    if not os.path.exists(TEST_OUTPUT_DIR):
        os.makedirs(TEST_OUTPUT_DIR)

    # --- summarize_target_distribution のテスト ---
    summarize_target_distribution(test_data, 'Perished')
    summarize_target_distribution(test_data, 'Sex')

    # --- visualize_feature_vs_target のテスト ---
    print("\n--- Visualizing features (test) ---")
    features_to_test_vis = ['Age', 'Fare', 'Pclass', 'Sex', 'Embarked']
    
    # プレフィックスなし
    for feature in features_to_test_vis:
        visualize_feature_vs_target(test_data, feature, 'Perished', output_dir=TEST_OUTPUT_DIR)
    
    # プレフィックスあり
    for feature in features_to_test_vis:
        visualize_feature_vs_target(test_data, feature, 'Perished', output_dir=TEST_OUTPUT_DIR, filename_prefix="testprefix_")

    # 存在しないカラムのテスト
    visualize_feature_vs_target(test_data, "NonExistentColumn", "Perished", output_dir=TEST_OUTPUT_DIR)

    # 目的変数が存在しないテスト
    visualize_feature_vs_target(test_data, "Age", "NonExistentTarget", output_dir=TEST_OUTPUT_DIR)

    # ターゲットがカテゴリカルでない場合の CountPlot (意図しない挙動になるか確認)
    # visualize_feature_vs_target(test_data, 'Sex', 'Age', output_dir=TEST_OUTPUT_DIR, filename_prefix="cat_target_issue_")

    print(f"\nTest visualizations saved in {TEST_OUTPUT_DIR}")
    print("Please check the directory for output files.") 