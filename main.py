import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("improved_dataset.csv")

# =========================
# CLEAN TOOLS COLUMN
# =========================
df["tools_and_technologies"] = df["tools_and_technologies"].apply(
    lambda x: [i.strip().lower() for i in str(x).split(",")]
)

# =========================
# MULTI LABEL ENCODING (SKILLS)
# =========================
mlb = MultiLabelBinarizer()
skills_encoded = mlb.fit_transform(df["tools_and_technologies"])
skills_df = pd.DataFrame(skills_encoded, columns=mlb.classes_)

# =========================
# ROLE ENCODING
# =========================
le = LabelEncoder()
df["role_encoded"] = le.fit_transform(df["role_name"])

# =========================
# TRAIN RANDOM FOREST MODEL
# =========================
X = skills_df
y = df["role_encoded"]

model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
model.fit(X, y)

print("Model trained successfully")

# =========================
# TF-IDF FOR SKILL RECOMMENDATION
# =========================
df["skills_text"] = df["tools_and_technologies"].apply(lambda x: " ".join(x))

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df["skills_text"])

# =========================
# USER INPUT
# =========================
name = input("\nEnter your name: ")

user_skills = input("Enter your skills (comma separated): ").lower().split(",")
user_skills = [s.strip() for s in user_skills]

preferred_companies = input("Enter preferred companies (comma separated): ").lower().split(",")
preferred_companies = [c.strip() for c in preferred_companies]

# =========================
# ROLE PREDICTION
# =========================
user_vector = pd.DataFrame(
    mlb.transform([user_skills]),
    columns=mlb.classes_
)

pred_role = model.predict(user_vector)
pred_role_name = le.inverse_transform(pred_role)[0]

print(f"\n{name}, you are best suited for '{pred_role_name}' role")

# =========================
# FILTER DATA BY COMPANIES
# =========================
filtered_df = df[df["company_name"].isin(preferred_companies)]

# fallback if no company match
if len(filtered_df) < 5:
    print("\nNot enough company-specific data, using global dataset...")
    filtered_df = df

# =========================
# TF-IDF ON FILTERED DATA
# =========================
filtered_text = filtered_df["tools_and_technologies"].apply(lambda x: " ".join(x))

filtered_tfidf = tfidf.transform(filtered_text)

# =========================
# USER VECTOR FOR SIMILARITY
# =========================
user_text = " ".join(user_skills)
user_tfidf = tfidf.transform([user_text])

similarity = cosine_similarity(user_tfidf, filtered_tfidf)

# =========================
# IMPROVED SKILL RECOMMENDATION
# =========================
similar_indices = similarity.argsort()[0][-15:]

skill_freq = {}

for idx in similar_indices:
    for skill in filtered_df.iloc[idx]["tools_and_technologies"]:
        if skill not in user_skills:
            skill_freq[skill] = skill_freq.get(skill, 0) + 1

recommended_skills = sorted(skill_freq, key=skill_freq.get, reverse=True)[:5]

print("\nRecommended skills to learn:")
for skill in recommended_skills:
    print("-", skill)
