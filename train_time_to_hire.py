# train_time_to_hire_model.py

import os
import ast
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

MAX_TRAIN_ROWS = 50000  # щоб не вбити оперативку

SOFT_SKILLS = {
    "комунікабельність",
    "стресостійкість",
    "відповідальність",
    "активність",
    "пунктуальність",
    "дисциплінованість",
    "стабільність",
    "організованість",
    "уважність",
    "бажання розвиватися",
    "креативність",
    "уміння працювати в команді",
    "робота в команді",
    "лідерські якості",
    "мотивація",
    "ініціативність",
    "самоорганізація",
    "наполегливість",
    "швидке навчання",
}


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data():
    load_dotenv()

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT"),
    )

    # time_to_hire — у днях, беремо тільки закриті вакансії з валідним значенням
    df = pd.read_sql(
        """
        SELECT job_id,
               title,
               description,
               skills,
               salary_average,
               category,
               location,
               time_to_hire
        FROM jobs
        WHERE time_to_hire IS NOT NULL
          AND time_to_hire > 0
        """,
        conn,
    )

    conn.close()
    return df


# ============================================================
# 2. SKILLS PARSING + CLEANING
# ============================================================

def parse_pg_skill_list(value):
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []

        if text.startswith("{") and text.endswith("}"):
            inner = text[1:-1]
            return [s.strip() for s in inner.split(",") if s.strip()]

        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass

        if "," in text:
            return [s.strip() for s in text.split(",") if s.strip()]

        return [text]

    return [str(value).strip()]


def clean_skills(skills):
    cleaned = []
    for s in skills:
        s_lower = str(s).lower().strip()
        if not s_lower:
            continue
        if s_lower in SOFT_SKILLS:
            continue
        if len(s_lower) < 2:
            continue
        cleaned.append(s_lower)
    return cleaned


def process_skills_column(df: pd.DataFrame) -> pd.DataFrame:
    df["skills_list"] = df["skills"].apply(parse_pg_skill_list)
    df["skills_list"] = df["skills_list"].apply(clean_skills)
    return df


# ============================================================
# 3. TOP SKILLS + ONE-HOT
# ============================================================

def build_top_skills(df: pd.DataFrame, top_n: int = 200):
    all_skills = []
    for skills in df["skills_list"]:
        all_skills.extend(skills)

    counter = Counter(all_skills)
    top = [skill for skill, _ in counter.most_common(top_n)]
    return top


def one_hot_encode_skills(df: pd.DataFrame, top_skills):
    rows = []
    for skills in df["skills_list"]:
        sset = set(skills)
        rows.append([1 if skill in sset else 0 for skill in top_skills])
    return np.array(rows, dtype=np.float32)


# ============================================================
# 4. TF-IDF (title + description)
# ============================================================

def build_text_features(df: pd.DataFrame):
    df["text"] = (
        df["title"].fillna("") + " " +
        df["description"].fillna("")
    )

    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        min_df=10,
    )

    X_text = vectorizer.fit_transform(df["text"])
    return X_text, vectorizer


# ============================================================
# 5. CATEGORY & LOCATION ENCODING
# ============================================================

def encode_category_location_train(df: pd.DataFrame):
    cat_encoder = LabelEncoder()
    loc_encoder = LabelEncoder()

    df["category"] = df["category"].astype(str).fillna("")
    df["location"] = df["location"].astype(str).fillna("")

    df["category_enc"] = cat_encoder.fit_transform(df["category"])
    df["location_enc"] = loc_encoder.fit_transform(df["location"])

    X_catloc = csr_matrix(
        df[["category_enc", "location_enc"]].values.astype(np.float32)
    )

    return X_catloc, cat_encoder, loc_encoder


# ============================================================
# 6. TRAIN MODEL
# ============================================================

def train_xgb(X, y):
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        objective="reg:squarederror",
        n_jobs=4,
    )
    model.fit(X, y)
    return model


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def main():
    print("📥 Loading data...")
    df = load_data()
    print(f"   Loaded rows with time_to_hire: {len(df)}")

    if df.empty:
        print("❌ No data with time_to_hire — nothing to train.")
        return

    print("🧹 Processing skills...")
    df = process_skills_column(df)

    # Семплінг для економії памʼяті
    if len(df) > MAX_TRAIN_ROWS:
        df = df.sample(MAX_TRAIN_ROWS, random_state=42).copy()
        print(f"   After sampling: {len(df)} rows")

    print("📊 Building top skills...")
    top_skills = build_top_skills(df, top_n=200)
    print(f"   Top skills: {len(top_skills)}")

    print("📦 Encoding skills...")
    X_sk = one_hot_encode_skills(df, top_skills)

    print("✍️ Building TF-IDF features...")
    X_text, vectorizer = build_text_features(df)

    print("🌍 Encoding category/location...")
    X_catloc, cat_encoder, loc_encoder = encode_category_location_train(df)

    print("💰 Preparing salary feature...")
    # salary_average як фіча; якщо десь None — замінимо на медіану
    salary = df["salary_average"].astype(float)
    salary_median = float(salary.median()) if not salary.isna().all() else 0.0
    salary_filled = salary.fillna(salary_median).values.astype(np.float32)
    X_salary = csr_matrix(salary_filled.reshape(-1, 1))

    print("🔗 Combining matrices...")
    X = hstack(
        [X_text, csr_matrix(X_sk), X_catloc, X_salary],
        format="csr",
    ).astype(np.float32)

    y = df["time_to_hire"].astype(np.float32).values

    print("🚀 Training model...")
    model = train_xgb(X, y)

    print("💾 Saving time-to-hire model & artifacts to models/ ...")
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/time_to_hire_model_xgb.pkl")
    joblib.dump(vectorizer, "models/time_to_hire_text_vectorizer.pkl")
    joblib.dump(top_skills, "models/time_to_hire_top_skills.pkl")
    joblib.dump(cat_encoder, "models/time_to_hire_category_encoder.pkl")
    joblib.dump(loc_encoder, "models/time_to_hire_location_encoder.pkl")
    joblib.dump(salary_median, "models/time_to_hire_salary_median.pkl")

    print("✅ DONE — time_to_hire model trained & artifacts saved.")


if __name__ == "__main__":
    main()
