import pandas as pd
import re
from fuzzywuzzy import fuzz

# === Step 1: Load and prepare data ===
df = pd.read_excel("conversations1.xlsx")
df.rename(columns={"Unnamed: 6": "Speaker", "Unnamed: 7": "Message"}, inplace=True)
df.reset_index(drop=True, inplace=True)

# === Step 2: Identify relevant AI questions ===
def is_allen_choice_prompt(msg):
    msg = str(msg).lower()
    return "deep dive" in msg and ("high level" in msg or "high-level" in msg)

choice_indices = df[
    (df["Speaker"] == "ALLEN") & df["Message"].apply(is_allen_choice_prompt)
].index

# === Step 3: Get next Guest User response for each prompt ===
response_records = []
for idx in choice_indices:
    ai_msg = str(df.at[idx, "Message"])
    for i in range(idx + 1, len(df)):
        if df.at[i, "Speaker"] == "Guest User":
            guest_msg = str(df.at[i, "Message"]).lower()
            response_records.append({
                "ai_message": ai_msg,
                "guest_message": guest_msg
            })
            break

# === Step 4: Classification Logic ===
def classify_by_prompt_match(ai_msg, guest_msg):
    guest_msg = guest_msg.lower()
    prompt_parts = re.split(r"\bor\b", ai_msg.lower())

    if len(prompt_parts) < 2:
        return "unclear"

    opt1 = prompt_parts[-2].strip()
    opt2 = prompt_parts[-1].strip()

    if "deep dive" in opt1:
        deep_opt = opt1
        high_opt = opt2
    elif "deep dive" in opt2:
        deep_opt = opt2
        high_opt = opt1
    else:
        return "unclear"

    # ✅ Rule 1: Direct match to user response
    if "deep dive" in guest_msg:
        return "deep dive"
    if "high level" in guest_msg or "high-level" in guest_msg:
        return "high level"

    # ✅ Rule 2: Fuzzy match to "pulling results" intent
    pulling_phrases = [
        "pull my results", "review remaining markers", "go over the rest",
        "show me everything", "full results", "go through all", 
        "check the rest", "look at all my data", "pull full panel",
        "rest of my numbers", "break everything down"
    ]
    for phrase in pulling_phrases:
        if fuzz.partial_ratio(guest_msg, phrase) >= 80:
            return "deep dive"

    # ✅ Rule 3: Fallback fuzzy match to prompt options
    score_deep = fuzz.partial_ratio(guest_msg, deep_opt)
    score_high = fuzz.partial_ratio(guest_msg, high_opt)

    if max(score_deep, score_high) < 50:
        return "unclear"
    return "deep dive" if score_deep > score_high else "high level"

# === Step 5: Apply classification ===
for r in response_records:
    r["label"] = classify_by_prompt_match(r["ai_message"], r["guest_message"])

# === Step 6: Show summary ===
counts = {"deep dive": 0, "high level": 0, "unclear": 0}
for r in response_records:
    counts[r["label"]] += 1

total = len(response_records)
print(f"\n📊 Classification Summary:")
print(f"Deep Dive:   {counts['deep dive']} ({counts['deep dive'] / total * 100:.1f}%)")
print(f"High Level:  {counts['high level']} ({counts['high level'] / total * 100:.1f}%)")
print(f"Unclear:     {counts['unclear']} ({counts['unclear'] / total * 100:.1f}%)")

# === Step 7: Print examples per category ===
def print_category(title, label):
    print(f"\n🔹 {title.upper()} RESPONSES:")
    for r in response_records:
        if r["label"] == label:
            print("AI Prompt:")
            print(f"→ {r['ai_message']}")
            print("Guest Response:")
            print(f"→ {r['guest_message']}\n")

print_category("high level", "high level")
print_category("deep dive", "deep dive")
print_category("unclear", "unclear")
