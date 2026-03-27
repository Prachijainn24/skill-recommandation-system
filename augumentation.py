import pandas as pd
import random

# load dataset
df = pd.read_csv("companies.csv")

# unique skills pool
all_skills = set()

for skills in df["tools_and_technologies"]:
    for s in skills.split(","):
        all_skills.add(s.strip())

all_skills = list(all_skills)

augmented_data = []

# number of rows to generate
target_size = 400

while len(augmented_data) < target_size:
    
    # pick random row
    row = df.sample(1).iloc[0].copy()

    # -------- MODIFY SALARY --------
    noise = random.uniform(-1.5, 1.5)
    row["salary"] = max(1, round(row["salary"] + noise, 2))

    # -------- MODIFY SKILLS --------
    skills = [s.strip() for s in row["tools_and_technologies"].split(",")]

    action = random.choice(["add", "remove"])

    if action == "add":
        new_skill = random.choice(all_skills)
        if new_skill not in skills:
            skills.append(new_skill)

    elif action == "remove" and len(skills) > 1:
        skills.remove(random.choice(skills))

    row["tools_and_technologies"] = ",".join(skills)

    augmented_data.append(row)

# create new dataframe
aug_df = pd.DataFrame(augmented_data)

# save file
aug_df.to_csv("augmented_dataset.csv", index=False)

print("Augmented dataset created with 400 rows")