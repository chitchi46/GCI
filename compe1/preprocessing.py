# GCI/compe1/preprocessing.py
from typing import List
import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


_TITLE_RE = re.compile(r",\s*([^\.]+)\.")
MAJOR = {"Capt", "Col", "Major", "Dr", "Rev"}
RARE  = {"Jonkheer", "Don", "Dona", "Lady", "Countess", "Sir"}


def _extract_title(name: str) -> str:
    m = _TITLE_RE.search(name)
    return m.group(1).strip() if m else "Unknown"


def _map_title(t: str) -> str:
    if t in {"Mlle", "Ms"}:
        return "Miss"
    if t == "Mme":
        return "Mrs"
    if t in MAJOR:
        return "Officer"
    if t in RARE:
        return "Royalty"
    return t


class TitanicPreprocessor(BaseEstimator, TransformerMixin):
    """
    - 欠損補完（train-fold平均を用いる）
    - 基本特徴量付与
    - OHE + Standardize
    """
    cat_cols_: List[str] = []
    num_cols_: List[str] = []

    # ---------------- fit ----------------
    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()

        # --- 欠損補完統計を覚える ---
        self.age_mean_  = df["Age"].mean()
        self.fare_mean_ = df["Fare"].mean()

        # --- Feature Engineering (train fold 専用) ---
        df = self._feature(df)

        # --- 列名を保持して ColumnTransformer を fit ---
        self.cat_cols_ = ["Sex", "Pclass", "Title", "Embarked",
                          "TicketPrefix", "Deck"]
        self.num_cols_ = ["Age", "SibSp", "Parch", "Fare", "FamilySize",
                          "TicketGroupSize", "FarePP", "IsAlone", "CabinMissing"]

        self.ct_ = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_cols_),
                ("num", StandardScaler(),                       self.num_cols_)
            ]
        ).fit(df[self.cat_cols_ + self.num_cols_])

        return self

    # ---------------- transform ----------------
    def transform(self, X: pd.DataFrame):
        df = X.copy()

        # 欠損補完
        df.loc[:, "Age"]      = df["Age"].fillna(self.age_mean_)
        df.loc[:, "Fare"]     = df["Fare"].fillna(self.fare_mean_)
        df.loc[:, "Embarked"] = df["Embarked"].fillna("S")

        # Feature Engineering
        df = self._feature(df)

        # OHE + 標準化
        return self.ct_.transform(df[self.cat_cols_ + self.num_cols_])

    # ---------------- private helpers ----------------
    def _feature(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Title
        out["Title"] = (
            out["Name"].apply(_extract_title).apply(_map_title)
        )

        # Family
        out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
        out["IsAlone"]    = (out["FamilySize"] == 1).astype(int)

        # Ticket
        out["TicketPrefix"] = (
            out["Ticket"]
            .str.replace(r"\d", "", regex=True)
            .str.replace(r"[\.\/]", "", regex=True)
            .str.upper()
            .fillna("_")
        )
        out["TicketGroupSize"] = out.groupby("Ticket")["Ticket"].transform("count")

        # Cabin / Deck
        out["Deck"] = out["Cabin"].fillna("U0").str[0]
        out["CabinMissing"] = out["Cabin"].isna().astype(int)

        # Fare per person
        out["FarePP"] = out["Fare"] / out["FamilySize"]

        return out
