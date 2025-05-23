# -*- coding: utf-8 -*-
import numpy as np
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
# config は直接は使用しないが、呼び出し側でパラメータが渡される想定

def train_model(X, y, X_test, params, n_splits, random_seed):
    """
    LightGBMモデルを学習し、予測値とCVスコアを返す関数
    """
    print("Starting model training...")
    # mlflow.lightgbm.autolog() # autologは呼び出し側(main.py)のrunコンテキスト内で行う方が適切

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    models = []
    cv_scores = []

    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        print(f"--- Fold {fold+1}/{n_splits} ---")
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])

        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_index] = val_preds
        test_preds += model.predict_proba(X_test)[:, 1] / n_splits
        models.append(model)

        val_preds_binary = (val_preds > 0.5).astype(int)
        fold_score = accuracy_score(y_val, val_preds_binary)
        cv_scores.append(fold_score)
        print(f"Fold {fold+1} Accuracy: {fold_score:.4f}")

    mean_cv_score = np.mean(cv_scores)
    print(f"Mean CV Accuracy: {mean_cv_score:.4f}")
    # mlflow.log_metric("mean_cv_accuracy", mean_cv_score) #これも呼び出し側で行う
    print("Model training finished.")
    return models, oof_preds, test_preds, mean_cv_score 