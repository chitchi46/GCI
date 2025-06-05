# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
# import mlflow.lightgbm # autolog を使わないので直接は不要になる可能性
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score # roc_auc_score を追加
import joblib # モデルのローカル保存のため
import os # パス操作のため
from pathlib import Path # パス操作のため
import matplotlib.pyplot as plt

def train_model(X, y, X_test, params, n_splits, random_seed, n_repeats=3): # parent_run_id は main.py の run context内で実行するため不要と判断
    """
    LightGBMモデルを学習し、予測値とCVスコアを返す関数
    (Nested Run 対応)
    """
    # mlflow.lightgbm.autolog(log_models=True, log_model_signatures=True, log_input_examples=True, log_datasets=False) # コメントアウト

    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_seed)

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    models = [] # 各foldのモデルを保持するリスト（これは従来通り）
    cv_scores = []
    cv_auc_scores = []
    cv_logloss_scores = []
    feature_importances = [] # 特徴量重要度を記録
    all_learning_curves = [] # 全foldの学習曲線を記録

    total_folds = n_splits * n_repeats

    # 各Foldのモデルを一時的に保存するディレクトリ (なければ作成)
    # このディレクトリは実行ごと、あるいはmain.pyの実験単位でクリーンアップすることが望ましい
    # ここでは実行のたびに作成・利用する
    project_root = Path(mlflow.get_artifact_uri()).parent.parent # ちょっと強引だが、mlrunsの一つ上をプロジェクトルートと仮定
    local_fold_model_dir = project_root / "temp_fold_models"
    local_fold_model_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_index, val_index) in enumerate(rkf.split(X, y)):
        fold_run_name = f"fold_{fold+1}_of_{total_folds}"
        # Nested Run を開始 (親Runのコンテキスト内で実行される)
        with mlflow.start_run(run_name=fold_run_name, nested=True) as child_run:
            print(f"Fold {fold+1}/{total_folds} training...")
            
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            # モデルの学習パラメータをログ
            mlflow.log_params(params) 
            mlflow.log_param("fold_number", fold + 1)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("val_size", len(X_val))
            # n_repeats, n_splits は親Runのparamsとして記録されるのでここでは不要かもしれない

            model = lgb.LGBMClassifier(**params)
            
            evaluation_results = {} # 結果を格納する辞書を初期化
            
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)], 
                      eval_metric=['binary_logloss'], # accuracyを削除してwarningを解消
                      callbacks=[
                          lgb.early_stopping(stopping_rounds=30, first_metric_only=False, verbose=False),
                          lgb.record_evaluation(evaluation_results) # 正しい関数名 record_evaluation に修正
                      ])

            # print(f"DEBUG: fold {fold+1} evaluation_results: {evaluation_results}") # DEBUG出力を削除

            # OOF予測 (確率)
            val_preds_proba = model.predict_proba(X_val)[:, 1]
            oof_preds[val_index] = val_preds_proba
            
            # テストデータ予測 (確率)
            if not X_test.empty:
                test_preds += model.predict_proba(X_test)[:, 1] / total_folds

            models.append(model) 

            # 検証データのバイナリ予測
            val_preds_binary = (val_preds_proba > 0.5).astype(int)
            
            # Foldスコア (Accuracy)
            fold_accuracy = accuracy_score(y_val, val_preds_binary)
            cv_scores.append(fold_accuracy) # これは親Runの平均CVスコア計算用
            mlflow.log_metric("validation_accuracy", fold_accuracy)
            
            # Foldスコア (AUC)
            fold_auc = roc_auc_score(y_val, val_preds_proba)
            cv_auc_scores.append(fold_auc)
            mlflow.log_metric("validation_auc", fold_auc)
            
            # Foldスコア (LogLoss)
            fold_logloss = log_loss(y_val, val_preds_proba)
            cv_logloss_scores.append(fold_logloss)
            mlflow.log_metric("validation_logloss", fold_logloss)
            
            # 特徴量重要度を記録
            fold_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_,
                'fold': fold + 1
            })
            feature_importances.append(fold_importance)
            
            # print(f"Fold {fold+1} Accuracy: {fold_accuracy:.4f}") # コメントアウト

            # Foldスコア (LogLoss) - record_evals から取得
            best_iter_idx = model.best_iteration_ -1 if model.best_iteration_ and model.best_iteration_ > 0 else len(evaluation_results['valid_0']['binary_logloss'])-1
            fold_logloss_eval = evaluation_results['valid_0']['binary_logloss'][best_iter_idx]
            mlflow.log_metric("validation_binary_logloss", fold_logloss_eval)
            
            # 学習曲線をログ - record_evals から取得（nested run）
            learning_curve = evaluation_results['valid_0']['binary_logloss']
            all_learning_curves.append(learning_curve)
            
            for i, logloss_val in enumerate(learning_curve):
                mlflow.log_metric("learning_curve_logloss", logloss_val, step=i)
            
            # accuracy メトリックは削除したので、学習曲線のログも削除

            # Foldモデルをローカルに一時保存
            fold_model_filename = f"model_fold_{fold+1}.joblib"
            fold_model_path = local_fold_model_dir / fold_model_filename
            joblib.dump(model, fold_model_path)
            
            # MLflowにモデルをアーティファクトとして記録 (joblica形式)
            # LightGBMの複雑な依存関係を避けるため、jobalibファイルをartifactとして保存
            mlflow.log_artifact(str(fold_model_path), f"model_fold_{fold+1}")
            # print(f"Fold {fold+1} model saved to MLflow Artifacts as 'model_fold_{fold+1}'")

    # 平均スコアの計算
    mean_cv_score = np.mean(cv_scores)
    mean_auc_score = np.mean(cv_auc_scores)
    mean_logloss_score = np.mean(cv_logloss_scores)
    
    # 特徴量重要度の平均を計算
    feature_importance_df = pd.concat(feature_importances, ignore_index=True)
    mean_importance = feature_importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
    
    # 特徴量重要度をMLflowにログ
    for feature, importance in mean_importance.items():
        mlflow.log_metric(f"feature_importance_{feature}", importance)
    
    # 特徴量重要度のプロットを作成してログ
    plt.figure(figsize=(10, 8))
    top_features = mean_importance.head(20)  # 上位20個の特徴量
    plt.barh(range(len(top_features)), top_features.values)
    plt.yticks(range(len(top_features)), top_features.index)
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Feature Importances (Average across folds)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # プロットをMLflowにログ
    plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
    mlflow.log_artifact("feature_importance.png")
    plt.close()
    
    # 親の実験に集約した学習曲線をログ
    if all_learning_curves:
        # 各foldの学習曲線の平均を計算
        max_iterations = min(len(curve) for curve in all_learning_curves)
        avg_learning_curve = []
        
        for i in range(max_iterations):
            epoch_values = [curve[i] for curve in all_learning_curves if i < len(curve)]
            avg_learning_curve.append(np.mean(epoch_values))
        
        # 平均学習曲線を親の実験にログ
        for i, avg_logloss in enumerate(avg_learning_curve):
            mlflow.log_metric("avg_learning_curve_logloss", avg_logloss, step=i)
        
        print(f"学習曲線の平均が記録されました（{max_iterations}イテレーション）")
        
        # 学習曲線の可視化プロットも作成
        plt.figure(figsize=(10, 6))
        
        # 各foldの学習曲線をプロット（薄い線）
        for i, curve in enumerate(all_learning_curves):
            epochs = range(1, len(curve) + 1)
            plt.plot(epochs, curve, alpha=0.3, color='gray', linewidth=1)
        
        # 平均学習曲線をプロット（太い線）
        avg_epochs = range(1, len(avg_learning_curve) + 1)
        plt.plot(avg_epochs, avg_learning_curve, color='red', linewidth=2, label='Average')
        
        plt.xlabel('Iteration')
        plt.ylabel('Binary LogLoss')
        plt.title('Learning Curves (All Folds + Average)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # プロットをMLflowにログ
        plt.savefig("learning_curves.png", dpi=150, bbox_inches='tight')
        mlflow.log_artifact("learning_curves.png")
        plt.close()
        
        print("学習曲線のプロットが保存されました")
    
    # 一時保存したモデル群を削除 (任意)
    # try:
    #     import shutil
    #     shutil.rmtree(local_fold_model_dir)
    #     print(f"Temporary folder '{local_fold_model_dir}' removed.")
    # except OSError as e:
    #     print(f"Error removing temporary folder '{local_fold_model_dir}': {e.strerror}")

    return models, oof_preds, test_preds, mean_cv_score, mean_auc_score, mean_logloss_score, feature_importance_df 