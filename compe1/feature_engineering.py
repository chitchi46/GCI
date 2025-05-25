# GCI/compe1/feature_engineering.py
import re
import pandas as pd

__all__ = ["add_basic_features"]

_TITLE_PATTERN = re.compile(r",\s*([^\.]+)\.")

# --- 1. Title 抽出 & 集約 -----------------
def _extract_title(name: str) -> str:
    m = _TITLE_PATTERN.search(name)
    return m.group(1).strip() if m else "Unknown"

def _map_title(t: str) -> str:
    major = {"Capt", "Col", "Major", "Dr", "Rev"}
    rare  = {"Jonkheer", "Don", "Dona", "Lady", "Countess", "Sir"}
    if t in {"Mlle", "Ms"}:        return "Miss"
    if t == "Mme":                 return "Mrs"
    if t in major:                 return "Officer"
    if t in rare:                  return "Royalty"
    return t

# --- 2. Family & Ticket 派生 --------------
def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Title
    out["Title"] = (
        out["Name"]
        .apply(_extract_title)
        .apply(_map_title)
    )

    # Family size + 状態
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"]    = (out["FamilySize"] == 1).astype(int)

    # Ticket prefix / group size
    out["TicketPrefix"] = (
        out["Ticket"]
        .str.replace(r"\d", "", regex=True)
        .str.replace(r"[\.\/]", "", regex=True)
        .str.upper()
        .fillna("NONE")
    )
    out["TicketGroupSize"] = (
        out.groupby("Ticket")["Ticket"].transform("count")
    )

    # Cabin デッキ & 欠損フラグ
    out["Deck"] = (
        out["Cabin"]
        .fillna("U0")
        .str[0]
    )
    out["CabinMissing"] = out["Cabin"].isna().astype(int)

    # Fare per person
    out["FarePP"] = out["Fare"] / out["FamilySize"]

    return out 