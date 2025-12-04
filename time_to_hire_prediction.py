import os
import ast
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# ============================================================
# 1. UNIVERSAL SKILL PARSER + CLEANING
# ============================================================

SOFT_SKILLS = {
    "комунікабельність", "стресостійкість", "відповідальність", "активність",
    "пунктуальність", "дисциплінованість", "стабільність", "організованість",
    "уважність", "бажання вчитися і розвиватися", "бажання вчитися",
    "бажання розвиватися", "креативність", "уміння працювати в команді",
    "робота в команді", "лідерські якості", "мотивація", "ініціативність",
    "самоорганізація", "наполегливість", "швидке навчання", "командна робота",
    "працьовитість", "доброзичливість"
}


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


# ============================================================
# 2. DB HELPERS
# ============================================================

def get_connection():
    load_dotenv()
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT"),
    )


def ensure_predicted_column():
    """Додаємо колонку predicted_time_to_hire, якщо її ще немає."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        ALTER TABLE jobs
        ADD COLUMN IF NOT EXISTS predicted_time_to_hire INTEGER;
    """)
    conn.commit()
    cur.close()
    conn.close()


def load_training_data():
    """Беремо вакансії, де time_to_hire вже відомий (закриті)."""
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


def load_prediction_data():
    """Беремо вакансії, де time_to_hire ще невідомий, але хочемо зробити прогноз."""
    conn = get_connection()
    df = pd.read_sql("""
        SELECT job_id, title, description, skills,
               salary_min, salary_max, salary_average,
               location, category, company,
               posted_date
        FROM jobs
        WHERE time_to_hire IS NULL
    """, conn)
    conn.close()
    return df


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

def build_top_skills(train_df, top_n=300):
    all_skills = []
    for raw in train_df["skills"]:
        parsed = parse_skill_list(raw)
        cleaned = clean_skill_list(parsed)
        all_skills.extend(cleaned)
    counter = Counter(all_skills)
    top = [sk for sk, _ in counter.most_common(top_n)]
    return top


def one_hot_skills(df, top_skills):
    rows = []
    for raw in df["skills"]:
        parsed = parse_skill_list(raw)
        cleaned = clean_skill_list(parsed)
        sset = set(cleaned)
        row = [1 if sk in sset else 0 for sk in top_skills]
        rows.append(row)
    return csr_matrix(np.array(rows))


def build_text_vectorizer(train_df):
    texts = (
            train_df["title"].fillna("") + " " +
            train_df["description"].fillna("")
    )
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=5
    )
    X_text = vectorizer.fit_transform(texts)
    return vectorizer, X_text


def transform_text(vectorizer, df):
    texts = (
            df["title"].fillna("") + " " +
            df["description"].fillna("")
    )
    return vectorizer.transform(texts)


def encode_categoricals(train_df, pred_df):
    """LabelEncoder для category, location, company, фітимо на об’єднаному датафреймі."""
    full = pd.concat([train_df[["category", "location", "company"]],
                      pred_df[["category", "location", "company"]]],
                     axis=0)

    cat_enc = LabelEncoder()
    loc_enc = LabelEncoder()
    comp_enc = LabelEncoder()

    cat_enc.fit(full["category"].astype(str))
    loc_enc.fit(full["location"].astype(str))
    comp_enc.fit(full["company"].astype(str))

    train_cat = cat_enc.transform(train_df["category"].astype(str))
    train_loc = loc_enc.transform(train_df["location"].astype(str))
    train_comp = comp_enc.transform(train_df["company"].astype(str))

    pred_cat = cat_enc.transform(pred_df["category"].astype(str))
    pred_loc = loc_enc.transform(pred_df["location"].astype(str))
    pred_comp = comp_enc.transform(pred_df["company"].astype(str))

    X_train_cat = csr_matrix(np.vstack([train_cat, train_loc, train_comp]).T)
    X_pred_cat = csr_matrix(np.vstack([pred_cat, pred_loc, pred_comp]).T)

    return X_train_cat, X_pred_cat


def add_numeric_features(df):
    """Створюємо додаткові числові фічі."""
    out = pd.DataFrame(index=df.index)

    # salary features
    out["salary_min"] = df["salary_min"].fillna(0).astype(float)
    out["salary_max"] = df["salary_max"].fillna(0).astype(float)
    out["salary_avg"] = df["salary_average"].fillna(0).astype(float)

    # posted_date → month, weekday, age (дні від сьогодні)
    today = datetime.today().date()
    posted = pd.to_datetime(df["posted_date"], errors="coerce").dt.date

    out["posted_month"] = pd.Series([d.month if d else 0 for d in posted], index=df.index)
    out["posted_weekday"] = pd.Series([d.weekday() if d else 0 for d in posted], index=df.index)
    out["age_days"] = pd.Series([(today - d).days if d else 0 for d in posted], index=df.index)

    # text length / skills count
    out["desc_len"] = df["description"].fillna("").apply(lambda x: len(str(x)))
    out["skills_count"] = df["skills"].apply(
        lambda x: len(clean_skill_list(parse_skill_list(x)))
    )

    return csr_matrix(out.values)


# ============================================================
# 4. TRAIN MODEL
# ============================================================

def train_time_to_hire_model():
    print("📥 Loading training and prediction data...")
    train_df = load_training_data()
    pred_df = load_prediction_data()

    if train_df.empty:
        print("❌ Немає жодної вакансії з заповненим time_to_hire — нема на чому вчитись.")
        return

    if pred_df.empty:
        print("ℹ Немає вакансій для прогнозу, але модель можна все одно натренувати.")

    # TOP SKILLS
    print("🧠 Building top skills...")
    top_skills = build_top_skills(train_df, top_n=300)
    print(f"   Взяли {len(top_skills)} найпопулярніших скілів.")

    # SKILLS ONE-HOT
    print("📦 One-hot skills...")
    X_train_sk = one_hot_skills(train_df, top_skills)
    X_pred_sk = one_hot_skills(pred_df, top_skills) if not pred_df.empty else csr_matrix((0, len(top_skills)))

    # TEXT TF-IDF
    print("✍️ TF-IDF text features...")
    vectorizer, X_train_text = build_text_vectorizer(train_df)
    X_pred_text = transform_text(vectorizer, pred_df) if not pred_df.empty else csr_matrix((0, X_train_text.shape[1]))

    # CATEGORICAL ENCODING
    print("🌍 Encoding categoricals...")
    X_train_cat, X_pred_cat = encode_categoricals(train_df, pred_df)

    # NUMERIC FEATURES
    print("📊 Numeric features...")
    X_train_num = add_numeric_features(train_df)
    X_pred_num = add_numeric_features(pred_df) if not pred_df.empty else csr_matrix((0, X_train_num.shape[1]))

    # COMBINE ALL
    print("🔗 Combining feature matrices...")
    X_train = hstack([X_train_text, X_train_sk, X_train_cat, X_train_num])
    X_pred = hstack([X_pred_text, X_pred_sk, X_pred_cat, X_pred_num]) if not pred_df.empty else None

    y_train = train_df["time_to_hire"].astype(float).values

    # TRAIN MODEL
    print("🚀 Training XGBoost regressor for time_to_hire...")
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        objective="reg:squarederror"
    )
    model.fit(X_train, y_train)

    return {
        "model": model,
        "top_skills": top_skills,
        "vectorizer": vectorizer,
    }, pred_df, X_pred


# ============================================================
# 5. PREDICT & SAVE TO DB
# ============================================================

def update_predictions_in_db(pred_df, preds):
    if pred_df.empty:
        print("ℹ Немає вакансій для оновлення прогнозу.")
        return

    conn = get_connection()
    cur = conn.cursor()

    for job_id, p in zip(pred_df["job_id"], preds):
        # обмежимо від 1 до 365 днів, щоб не було дичі
        days = int(max(1, min(365, round(p))))
        cur.execute("""
            UPDATE jobs
            SET predicted_time_to_hire = %s
            WHERE job_id = %s;
        """, (days, job_id))

    conn.commit()
    cur.close()
    conn.close()
    print("💾 Прогнози записано в jobs.predicted_time_to_hire")


# ============================================================
# 6. MAIN
# ============================================================

def main():
    ensure_predicted_column()

    artifacts, pred_df, X_pred = train_time_to_hire_model()
    if artifacts is None:
        return

    model = artifacts["model"]

    if X_pred is not None and X_pred.shape[0] > 0:
        print("🤖 Predicting time_to_hire for open/unknown vacancies...")
        preds = model.predict(X_pred)
        update_predictions_in_db(pred_df, preds)

        print("🔥 Приклади прогнозів:")
        for job_id, p in list(zip(pred_df["job_id"], preds))[:10]:
            print(f"  job_id={job_id} → {round(p, 1)} днів")
    else:
        print("ℹ Немає вакансій із NULL time_to_hire — тільки модель натренували.")

    print("✅ Done.")


if __name__ == "__main__":
    main()
