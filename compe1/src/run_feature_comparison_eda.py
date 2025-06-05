import pandas as pd
# import os # 削除 (pathlib を使う)
# import sys # 削除
from pathlib import Path # 追加

# プロジェクトルートとcompe1ルートをsys.pathに追加 # 削除
# _CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # 削除
# _COMPE1_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..')) # 削除
# _PROJECT_ROOT = os.path.abspath(os.path.join(_COMPE1_ROOT, '..')) # 削除

# if _PROJECT_ROOT not in sys.path: # 削除
#     sys.path.insert(0, _PROJECT_ROOT) # 削除
# if _COMPE1_ROOT not in sys.path: # 主に compe1.src のインポートのため # 削除
#     sys.path.insert(0, _COMPE1_ROOT) # 削除

from src import config # 変更
from src.data_loader import load_train_data, load_test_data # 変更
from src.preprocessor import preprocess_data # 変更
from src.eda import visualize_feature_distribution_comparison # 変更
from src.utils import get_project_root # 追加

PROJECT_ROOT = get_project_root() # 追加

def compare_feature_distributions():
    print("Starting feature distribution comparison EDA...")

    # 1. データ読み込み
    print(f"Loading data from: {PROJECT_ROOT / config.TRAIN_DATA_PATH} and {PROJECT_ROOT / config.TEST_DATA_PATH}") # 修正
    train_df_orig = load_train_data(PROJECT_ROOT / config.TRAIN_DATA_PATH) # 修正
    test_df_orig = load_test_data(PROJECT_ROOT / config.TEST_DATA_PATH)   # 修正

    if train_df_orig is None or test_df_orig is None:
        print("Failed to load data. Exiting.")
        return

    # 2. データ前処理
    print("\nRunning preprocessing...")
    # preprocess_data は (X, y, X_test) を返す. y はここでは不要
    # IDカラムはpreprocess_dataの出力に含まれていない想定
    X_processed, _, X_test_processed = preprocess_data(train_df_orig.copy(), test_df_orig.copy())
    print("Preprocessing finished.")
    print(f"Shape of processed train features: {X_processed.shape}")
    print(f"Shape of processed test features: {X_test_processed.shape}")

    # 出力ディレクトリの準備
    # output_plot_dir = os.path.join(_COMPE1_ROOT, config.OUTPUT_DIR, "eda_plots") # 削除
    output_plot_dir_path = PROJECT_ROOT / config.OUTPUT_DIR / "eda_plots" # 追加
    # if not os.path.exists(output_plot_dir): # 削除
    #     os.makedirs(output_plot_dir) # 削除
    output_plot_dir_path.mkdir(parents=True, exist_ok=True) # 追加
    print(f"Created directory: {output_plot_dir_path}") # 修正

    features_to_compare = ['Age', 'Fare']

    for feature in features_to_compare:
        print(f"\n--- Comparing distributions for feature: {feature} ---")
        if feature not in X_processed.columns:
            print(f"Feature '{feature}' not found in processed training data. Skipping.")
            continue
        if feature not in X_test_processed.columns:
            print(f"Feature '{feature}' not found in processed test data. Skipping.")
            continue

        train_series = X_processed[feature]
        test_series = X_test_processed[feature]

        # 3. 可視化による比較
        visualize_feature_distribution_comparison(
            train_series=train_series,
            test_series=test_series,
            feature_name=feature,
            output_dir=str(output_plot_dir_path), # 修正
            filename_prefix=f"comparison_{feature.lower()}_"
        )

        # 4. 統計量の比較
        print(f"\nStatistics for {feature} in Training Data:")
        print(train_series.describe())
        print(f"\nStatistics for {feature} in Test Data:")
        print(test_series.describe())

    print("\nFeature distribution comparison EDA finished.")
    print(f"Plots saved in: {output_plot_dir_path}") # 修正

if __name__ == '__main__':
    compare_feature_distributions() 