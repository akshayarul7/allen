import pandas as pd
import requests
import time
from collections import Counter

# === CONFIG ===
API_KEY = "AIzaSyBh2VrsCtsgT-7qW9QsLOF3JszshO0cjqM"
INPUT_FILE = "conversations1.xlsx"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
HEADERS = {"Content-Type": "application/json"}
BATCH_SIZE = 10
DELAY_BETWEEN_CALLS = 2  # seconds

PROMPT_TEMPLATE = """
You are a classification assistant. Given the following numbered conversation logs between a health AI and guests, assign each log to a meaningful topic category (e.g., Supplement Questions, Symptom Discussion, Lab Result Clarification, etc).

Respond with a numbered list like:
1. Topic
2. Topic
3. Topic
...

Only return the list of topics. Do not include any other text.

Conversation Logs:
{batched_logs}
"""

def classify_logs_batch(batched_prompt):
    payload = {
        "contents": [{"parts": [{"text": PROMPT_TEMPLATE.format(batched_logs=batched_prompt)}]}]
    }
    params = {"key": API_KEY}

    try:
        response = requests.post(API_URL, headers=HEADERS, params=params, json=payload)
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"❌ Error: {e}")
        return "Error"

df = pd.read_excel(INPUT_FILE)
df.rename(columns={"Unnamed: 6": "Speaker", "Unnamed: 7": "Message"}, inplace=True)
df = df[["Speaker", "Message"]].dropna()

logs = []
current_log = []

for _, row in df.iterrows():
    if row["Speaker"] == "ALLEN" and current_log:
        logs.append("\n".join(current_log))
        current_log = []
    current_log.append(f"{row['Speaker']}: {row['Message']}")

if current_log:
    logs.append("\n".join(current_log))

def format_batch(logs, start_idx):
    formatted = []
    for i, log in enumerate(logs):
        formatted.append(f"{start_idx + i + 1}. {log}")
    return "\n\n".join(formatted)

all_topics = []
for i in range(0, len(logs), BATCH_SIZE):
    batch_logs = logs[i:i+BATCH_SIZE]
    formatted_logs = format_batch(batch_logs, i)

    result = classify_logs_batch(formatted_logs)
    for line in result.splitlines():
        topic = line.split(".", 1)[-1].strip()
        all_topics.append(topic)

    time.sleep(DELAY_BETWEEN_CALLS)

topic_counts = Counter(all_topics)
print("\n📊 Top 10 Most Common Topics:")
for topic, count in topic_counts.most_common(10):
    print(f"{topic:<30} {count}")
