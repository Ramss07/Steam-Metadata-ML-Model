import pandas as pd
import numpy as np
from pathlib import Path
from ast import literal_eval
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ==== SETTINGS I CAN TWEAK ====
DATA_PATH = Path("steam_games_cleaned.csv")
RANDOM_STATE = 42
POSITIVE_THRESHOLD = 75.0  # if a game is at or above this %, I call it "positive"
MIN_REVIEWS = 10           # only keep games with at least this many total reviews

sns.set(style="whitegrid")


def parse_list_cell(x):
    """
    A lot of columns are saved as strings that look like lists, like:
        "['Single-player', 'Co-op']"
    This turns those into real Python lists. If it's empty or weird, I just return [].
    """
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        x = x.strip()
        if x == "" or x == "[]":
            return []
        try:
            val = literal_eval(x)
            if isinstance(val, list):
                return val
        except (ValueError, SyntaxError):
            return []
    return []


def multi_hot_encode(df, col, prefix=None, top_n=None):
    """
    Turn a list column into a bunch of 0/1 columns.
    Example: genres -> genre_action, genre_rpg, etc.
    If top_n is set, I only keep the top_n most common items.
    """
    prefix = prefix or col

    # count how often each thing shows up
    counter = Counter()
    for items in df[col]:
        for item in items:
            counter[item] += 1

    # pick which items we want to turn into columns
    if top_n is not None:
        items = [item for item, _ in counter.most_common(top_n)]
    else:
        items = list(counter.keys())

    print(f"Multi-hot encoding '{col}' with {len(items)} unique values.")

    # make one column per item (1 if the row has it, 0 otherwise)
    for item in items:
        safe_name = (
            f"{prefix}_{item}"
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
            .lower()
        )
        df[safe_name] = df[col].apply(lambda lst: int(item in lst))

    return df


def main():
    print(f"Loading cleaned data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print("Initial shape:", df.shape)

    # ====== BASIC FILTERS AND LABEL ======

    # I only want games with at least MIN_REVIEWS, so labels are less noisy
    if "total_reviews" not in df.columns:
        raise ValueError("total_reviews column not found in cleaned CSV.")
    before = df.shape[0]
    df = df[df["total_reviews"] >= MIN_REVIEWS]
    print(f"Filtered games with < {MIN_REVIEWS} reviews: {before} -> {df.shape[0]} rows")

    # make a simple 0/1 label based on positive_percentual
    if "positive_percentual" not in df.columns:
        raise ValueError("positive_percentual not found for creating label.")

    df["label_positive"] = (df["positive_percentual"] >= POSITIVE_THRESHOLD).astype(int)
    y = df["label_positive"].values

    # ====== CLEAN UP LIST-LIKE COLUMNS ======

    list_cols = ["developers", "publishers", "categories", "genres", "platforms"]
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_list_cell)
        else:
            print(f"Warning: list-like column '{col}' not found in df.")

    # simple numeric info: how many devs / publishers worked on it
    if "developers" in df.columns:
        df["num_developers"] = df["developers"].apply(len)
    else:
        df["num_developers"] = 0

    if "publishers" in df.columns:
        df["num_publishers"] = df["publishers"].apply(len)
    else:
        df["num_publishers"] = 0

    # ====== TURN CATEGORIES / GENRES / PLATFORMS INTO 0/1 COLUMNS ======

    if "categories" in df.columns:
        df = multi_hot_encode(df, "categories", prefix="cat")
    if "genres" in df.columns:
        df = multi_hot_encode(df, "genres", prefix="genre")
    if "platforms" in df.columns:
        df = multi_hot_encode(df, "platforms", prefix="plat")

    # ====== BASIC NUMERIC COLUMNS ======

    # is_free as 0/1
    if "is_free" in df.columns:
        if df["is_free"].dtype == "object":
            df["is_free"] = (
                df["is_free"]
                .astype(str)
                .str.strip()
                .str.upper()
                .map({"TRUE": 1, "FALSE": 0})
                .fillna(0)
                .astype(int)
            )
        else:
            df["is_free"] = df["is_free"].astype(int)
    else:
        df["is_free"] = 0

    # metacritic: treat 0 as "missing", fill with median of non-zero values
    if "metacritic" in df.columns:
        non_zero = df["metacritic"].replace(0, np.nan)
        median_mc = non_zero.median()
        df["metacritic"] = non_zero.fillna(median_mc).astype(float)
    else:
        df["metacritic"] = 0.0

    # release_year: fill missing with the median year
    if "release_year" in df.columns:
        median_year = df["release_year"].median()
        df["release_year"] = df["release_year"].fillna(median_year).astype(int)
    else:
        df["release_year"] = 0

    # ====== BUILD FEATURE MATRIX (X) ======

    # I don't want anything that directly uses reviews in the input features
    drop_cols = [
        "label_positive",      # my target
        "positive_percentual", # used to make the label
        "total_reviews",
        "total_positive",
        "total_negative",
        "developers",
        "publishers",
        "categories",
        "genres",
        "platforms",
        "is_released",         # all True in cleaned CSV
    ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    print("Feature matrix shape before type check:", X.shape)

    # just in case any object columns slipped through, drop them
    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        print("Warning: These columns are still object dtype and will be dropped:")
        print(obj_cols)
        X = X.drop(columns=obj_cols)

    print("Final feature matrix shape:", X.shape)

    # ====== TRAIN / TEST SPLIT ======

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Train shape:", X_train.shape, " Test shape:", X_test.shape)

    # ====== LOGISTIC REGRESSION ======

    # scale features for logistic regression so nothing dominates just because of scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n=== Training Logistic Regression (metadata only) ===")
    log_reg = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
    )
    log_reg.fit(X_train_scaled, y_train)

    y_pred_lr = log_reg.predict(X_test_scaled)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"Logistic Regression Accuracy: {acc_lr:.4f}")
    print("\nLogistic Regression Classification Report:")
    print(classification_report(y_test, y_pred_lr, digits=4))

    print("Logistic Regression Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_lr))

    # ====== RANDOM FOREST ======

    print("\n=== Training Random Forest (metadata only) ===")
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Accuracy: {acc_rf:.4f}")
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf, digits=4))

    print("Random Forest Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))

    # ====== FEATURE IMPORTANCE FROM RANDOM FOREST ======

    importances = rf.feature_importances_
    feature_names = X.columns.values

    # print top 30 features in the console
    print("\nTop 30 Most Important Features (Random Forest, metadata only):")
    topn = 30
    indices_top = np.argsort(importances)[::-1][:topn]
    for rank, idx in enumerate(indices_top, start=1):
        print(f"{rank:2d}. {feature_names[idx]}: {importances[idx]:.4f}")

    # full ranking if I ever want to see everything
    print("\n=== Full Feature Importance Ranking (All Features) ===")
    sorted_features = sorted(zip(importances, feature_names), reverse=True)
    for rank, (imp, name) in enumerate(sorted_features, start=1):
        print(f"{rank:3d}. {name}: {imp:.4f}")

    # =========================
    #      VISUALIZATIONS
    # =========================

    # ---------------------------------------------------------
    # REVIEW POSITIVITY VS REVIEW COUNT (Original Version)
    # Now with a clear explanation at the bottom of the plot
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 6))

    # Basic scatter plot
    sns.scatterplot(
        data=df,
        x=np.log1p(df["total_reviews"]),
        y="positive_percentual",
        alpha=0.25,
        s=12,
        edgecolor=None
    )

    # Simple linear regression line (what you originally had)
    sns.regplot(
        data=df,
        x=np.log1p(df["total_reviews"]),
        y="positive_percentual",
        scatter=False,
        color="red",
        line_kws={"linewidth": 2}
    )

    plt.xlabel("log(total_reviews)")
    plt.ylabel("positive_percentual")
    plt.title("Review Positivity vs Review Count (Log Scale)")

    # ===== Add explanation text at the bottom =====
    explanation = (
        "What this shows:\n"
        "- Games with very few reviews (left side) can appear extremely good or extremely bad.\n"
        "  This is because a tiny number of reviews can swing the positivity percentage wildly.\n"
        "- As total reviews increase, scores become more stable and cluster around ~70–90%.\n"
        "- Only games with broad appeal reach high review counts, which pushes their scores\n"
        "  toward the middle and reduces extreme ratings."
    )

    plt.figtext(
        0.5, -0.15, explanation,
        wrap=True, ha="center", fontsize=10
    )

    plt.tight_layout()
    plt.savefig("viz_pos_vs_reviewcount_original_with_text.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: viz_pos_vs_reviewcount_original_with_text.png")

    # ---------------------------------------------------------
    # 2. TOP 30 FEATURE IMPORTANCES (RANDOM FOREST)
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 8))
    names_top = [feature_names[i] for i in indices_top]
    scores_top = importances[indices_top]

    plt.barh(names_top[::-1], scores_top[::-1])
    plt.title("Top 30 Most Important Features (Random Forest)")
    plt.xlabel("Feature Importance")
    plt.tight_layout()
    plt.savefig("viz_top30_feature_importances.png", dpi=300)
    plt.close()
    print("Saved: viz_top30_feature_importances.png")


if __name__ == "__main__":
    main()
