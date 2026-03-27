import pandas as pd
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity

print("Model starting...")

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv("augmented_dataset.csv")

# -------------------------------
# 2. CLEAN DATA
# -------------------------------
df = df.drop(["company_id"], axis=1)

df["tools_and_technologies"] = df["tools_and_technologies"].apply(
    lambda x: [i.strip().lower() for i in x.split(",")]
)

df["company_name"] = df["company_name"].str.lower()

# -------------------------------
# 3. ENCODE SKILLS
# -------------------------------
mlb = MultiLabelBinarizer()

skills_encoded = pd.DataFrame(
    mlb.fit_transform(df["tools_and_technologies"]),
    columns=mlb.classes_
)

df = pd.concat([df, skills_encoded], axis=1)

# -------------------------------
# 4. ENCODE COMPANY
# -------------------------------
company_encoded = pd.get_dummies(df["company_name"])

# -------------------------------
# 5. COMBINE FEATURES
# -------------------------------
X = pd.concat([skills_encoded, company_encoded], axis=1)
y = df["role_name"]

# -------------------------------
# 6. TRAIN MODEL
# -------------------------------
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

print("Model trained successfully")

# -------------------------------
# 7. ROLE-SKILL MATRIX
# -------------------------------
role_skill_df = df.groupby("role_name")[mlb.classes_].sum()

# -------------------------------
# 8. USER INPUT
# -------------------------------
name = input("\nEnter your name: ")
skills_input = input("Enter your skills (comma separated): ")
companies_input = input("Enter preferred companies (comma separated): ")

student_skills = [s.strip().lower() for s in skills_input.split(",")]
companies = [c.strip().lower() for c in companies_input.split(",")]

# -------------------------------
# 9. CONVERT INPUT TO VECTOR
# -------------------------------
input_vector = np.zeros(len(X.columns))

# encode skills
for skill in student_skills:
    if skill in mlb.classes_:
        idx = list(mlb.classes_).index(skill)
        input_vector[idx] = 1

# encode companies
for company in companies:
    if company in company_encoded.columns:
        idx = len(mlb.classes_) + list(company_encoded.columns).index(company)
        input_vector[idx] = 1

input_df = pd.DataFrame([input_vector], columns=X.columns)

# -------------------------------
# 10. PREDICT ROLE (Random Forest)
# -------------------------------
predicted_role = model.predict(input_df)[0]

print(f"\n{name}, you are best suited for '{predicted_role}' role")

# -------------------------------
# 11. FILTER DATA (ROLE + COMPANIES)
# -------------------------------
filtered_df = df[
    (df["role_name"] == predicted_role) &
    (df["company_name"].isin(companies))
]

# fallback
if filtered_df.empty:
    filtered_df = df[df["role_name"] == predicted_role]

# -------------------------------
# 12. BUILD ROLE-SKILL MATRIX (FILTERED)
# -------------------------------
filtered_role_skill = filtered_df.groupby("role_name")[mlb.classes_].sum()

# -------------------------------
# 13. COSINE SIMILARITY (CORE LOGIC)
# -------------------------------

# only compare skill part
user_skill_vector = input_vector[:len(mlb.classes_)]

similarities = cosine_similarity(
    [user_skill_vector],
    filtered_role_skill
)[0]

# get most similar role
best_index = similarities.argmax()
best_role = filtered_role_skill.index[best_index]

print(f"Most similar role based on your skills: {best_role}")

# -------------------------------
# 14. RECOMMEND SKILLS
# -------------------------------
role_vector = filtered_role_skill.loc[best_role]

# remove generic skills
bad_skills = ["backend", "frontend", "data", "analytics", "product", "storage"]

# sort by importance
sorted_skills = sorted(
    role_vector.items(),
    key=lambda x: x[1],
    reverse=True
)

recommended = [
    skill for skill, val in sorted_skills
    if val > 2
    and skill not in student_skills
    and skill not in bad_skills
]

print("\nRecommended skills to learn:")
for skill in recommended[:5]:
    print("-", skill)