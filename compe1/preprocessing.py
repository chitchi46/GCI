# GCI/compe1/preprocessing.py
from typing import Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from feature_engineering import add_basic_features

__all__ = ["prepare_train_test"]

CATEGORICAL = ["Sex", "Pclass", "Title", "Embarked", "TicketPrefix", "Deck"]
NUMERIC     = ["Age", "SibSp", "Parch", "Fare", "FamilySize",
               "TicketGroupSize", "FarePP", "IsAlone", "CabinMissing"]

def _build_ct() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", StandardScaler(),                       NUMERIC),
        ],
        remainder="passthrough" # Keep other columns, if any
    )

def prepare_train_test(
    df_train_raw: pd.DataFrame,
    df_test_raw: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    # ----- 1. 欠損補完 （train の統計で fit） ----------------
    train = df_train_raw.copy()
    test  = df_test_raw.copy()

    age_mean  = train["Age"].mean()
    fare_mean = train["Fare"].mean()

    for df in (train, test):
        df["Age"].fillna(age_mean, inplace=True)
        df["Fare"].fillna(fare_mean, inplace=True)
        df["Embarked"].fillna("S", inplace=True)

    # Cabin は FE 内で取り扱うので残す
    # ----- 2. Feature Engineering ---------------------------
    train_fe = add_basic_features(train)
    test_fe  = add_basic_features(test)

    # ----- 3. 作表  -----------------------------------------
    y_train = train_fe["Perished"]
    # Select only relevant columns for X_train before encoding
    X_train_cols = CATEGORICAL + NUMERIC
    X_train = train_fe[X_train_cols]
    X_test  = test_fe[X_train_cols] # Use same columns for test set

    # ----- 4. ColumnTransformer (リークなし) ------------------
    ct = _build_ct()
    # Pipeline expects a list of (name, transform) tuples.
    # The ColumnTransformer is already a transformer.
    # We can fit and transform directly with ct.

    X_train_transformed = ct.fit_transform(X_train)
    X_test_transformed  = ct.transform(X_test)

    # Get feature names after transformation
    # Access the fitted OneHotEncoder to get feature names for categorical columns
    feature_names_cat = ct.named_transformers_['cat'].get_feature_names_out(CATEGORICAL)
    # Combine with numeric column names (StandardScaler doesn't change names)
    feature_names = list(feature_names_cat) + NUMERIC

    X_train_enc = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
    X_test_enc  = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)

    return X_train_enc, y_train, X_test_enc 