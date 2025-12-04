# train_time_to_hire.py

import os
import ast
from collections import Counter
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor

SOFT_SKILLS = {
    "комунікабельність", "стресостійкість", "відповідальність", "активність",
    "пунктуальність", "дисциплінованість", "стабільність", "організованість",
    "уважність", "бажання вчитися і розвиватися", "бажання вчитися",
    "бажання розвиватися", "креативність", "уміння працювати в команді",
    "робота в команді", "лідерські якості", "мотивація", "ініціативність",
    "самоорганізація", "наполегливість", "швидке навчання", "командна робота",
    "працьовитість", "доброзичливість"
}


def get_connection():
    load_dotenv()
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT"),
    )


def parse_skill_list(value):
    """Універсальний парсер skills для форматів:
    - {a,b,c}
    - ['a','b']
    - "[\"a\", \"b\"]"
    - "a, b, c"
    - один skill як строка
    """
    if value is None:
        return []

    # вже Python list
    if isinstance(value, list):
        return [str(x).strip() for x in value]

    if isinstance(value, str):
        text = value.strip()

        # рядок, схожий на список
        if (text.startswith("[") and text.endswith("]")):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed]
            except Exception:
                pass

        # PostgreSQL масив {a,b,c}
        if text.startswith("{") and text.endswith("}"):
            inner = text[1:-1]
            return [s.strip() for s in inner.split(",") if s.strip()]

        # просто "a, b, c"
        if "," in text:
            return [s.strip() for s in text.split(",") if s.strip()]

        # fallback — один skill
        return [text]

    # інші типи
    return [str(value).strip()]


def clean_skill(skill: str):
    if not skill:
        return None
    s = skill.strip().lower()
    if len(s) < 2:
        return None
    if s in SOFT_SKILLS:
        return None
    return s


def clean_skill_list(skills):
    cleaned = []
    for s in skills:
        cs = clean_skill(s)
        if cs:
            cleaned.append(cs)
    return cleaned


def load_training_data():
    conn = get_connection()
    df = pd.read_sql("""
        SELECT job_id, title, description, skills,
               salary_min, salary_max, salary_average,
               location, category, company,
               posted_date, time_to_hire
        FROM jobs
        WHERE time_to_hire IS NOT NULL
          AND time_to_hire > 0
    """, conn)
    conn.close()
    return df


def build_top_skills(train_df, top_n=300):
    all_skills = []
    for raw in train_df["skills"]:
        parsed = parse_skill_list(raw)
        cleaned = clean_skill_list(parsed)
        all_skills.extend(cleaned)
    counter = Counter(all_skills)
    return [sk for sk, _ in counter.most_common(top_n)]


def one_hot_skills(df, top_skills):
    rows = []
    for raw in df["skills"]:
        parsed = parse_skill_list(raw)
        cleaned = clean_skill_list(parsed)
        sset = set(cleaned)
        rows.append([1 if sk in sset else 0 for sk in top_skills])
    return csr_matrix(np.array(rows))


def add_numeric_features(df):
    out = pd.DataFrame(index=df.index)

    out["salary_min"] = df["salary_min"].fillna(0).astype(float)
    out["salary_max"] = df["salary_max"].fillna(0).astype(float)
    out["salary_avg"] = df["salary_average"].fillna(0).astype(float)

    today = datetime.today().date()
    posted = pd.to_datetime(df["posted_date"], errors="coerce").dt.date

    out["age_days"] = pd.Series([(today - d).days if d else 0 for d in posted], index=df.index)
    out["desc_len"] = df["description"].fillna("").apply(lambda x: len(str(x)))
    out["skills_count"] = df["skills"].apply(
        lambda x: len(clean_skill_list(parse_skill_list(x)))
    )

    return csr_matrix(out.values)


def build_text_vectorizer(train_df):
    texts = (train_df["title"].fillna("") + " " + train_df["description"].fillna(""))
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5)
    X_text = vectorizer.fit_transform(texts)
    return vectorizer, X_text


def main():
    os.makedirs("models", exist_ok=True)

    print("📥 Loading training data...")
    train_df = load_training_data()
    if train_df.empty:
        print("❌ No training data with time_to_hire")
        return

    print("🧠 Building top skills...")
    top_skills = build_top_skills(train_df, top_n=300)

    print("📦 Skills one-hot...")
    X_sk = one_hot_skills(train_df, top_skills)

    print("✍️ Text TF-IDF...")
    vectorizer, X_text = build_text_vectorizer(train_df)

    print("📊 Numeric features...")
    X_num = add_numeric_features(train_df)

    print("🔗 Combine features...")
    X_train = hstack([X_text, X_sk, X_num])
    y_train = train_df["time_to_hire"].astype(float).values

    print("🚀 Train XGBoost...")
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)

    print("💾 Saving artifacts...")
    joblib.dump(model, "models/time_to_hire_xgb.pkl")
    joblib.dump(vectorizer, "models/time_to_hire_tfidf.pkl")
    joblib.dump(top_skills, "models/time_to_hire_top_skills.pkl")

    print("✅ Done training.")


if __name__ == "__main__":
    main()
