import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
from collections import Counter

# Load and rename
df = pd.read_excel("conversations2.xlsx")  
df.rename(columns={"Unnamed: 6": "Speaker", "Unnamed: 7": "Message"}, inplace=True)

# Question detection
question_starters = ("what", "why", "how", "should", "can", "do", "does", "is", "are", "could", "would", "will", "who", "when", "where", "which")
def is_question(msg):
    msg = str(msg).strip().lower()
    return "?" in msg or msg.startswith(question_starters)

# Filter Guest User questions
question_df = df[
    (df["Speaker"] == "Guest User") &
    (df["Message"].apply(is_question))
].copy()

# Clean question text
def clean_question(q):
    q = str(q).strip().lower()
    q = re.sub(r'[^\w\s]', '', q)
    q = re.sub(r'\s+', ' ', q)
    return q

question_df["Cleaned"] = question_df["Message"].apply(clean_question)
questions = question_df["Cleaned"].drop_duplicates().tolist()

# Embed and find paraphrases
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
embeddings = model.encode(questions, convert_to_tensor=True)
pairs = util.paraphrase_mining_embeddings(embeddings)

# Grouping paraphrased questions
threshold = 0.85
seen = set()
faq_groups = []

for score, i, j in pairs:
    if score < threshold:
        break
    q1, q2 = questions[i], questions[j]
    if q1 not in seen and q2 not in seen:
        faq_groups.append([q1, q2])
        seen.update([q1, q2])
    else:
        for group in faq_groups:
            if q1 in group or q2 in group:
                group.extend([q for q in [q1, q2] if q not in group])
                seen.update([q1, q2])
                break

# Count group sizes and sort
faq_groups = sorted(faq_groups, key=lambda g: len(g), reverse=True)
top_10_groups = faq_groups[:10]

# Print representative + variants
for idx, group in enumerate(top_10_groups, 1):
    print(f"\nFAQ #{idx} — Representative: {group[0]}")
    print("Variants:")
    for q in group[1:]:
        print(f" - {q}")
    print(f"[{len(group)} similar questions]")