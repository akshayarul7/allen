import pandas as pd
import spacy
from collections import Counter, defaultdict



df = pd.read_excel("/content/conversations.xlsx")

column_dictionary = {'Ways2Well - 05/20/2025 00:00 to 05/20/2025 23:59': 'Session ID', "Unnamed: 1": "Avatar Id", "Unnamed: 2": "Log #", "Unnamed: 3": "Date", "Unnamed: 4": "Avatar Title", "Unnamed: 5": "Talk Time", "Unnamed: 6":"Speaker", "Unnamed: 7": "Message"}
df.rename(columns = column_dictionary, inplace= True)


#pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz


nlp = spacy.load("en_ner_bc5cdr_md")
doc = nlp("Magnesium helps reduce glucose levels and boost testosterone.")
print([(ent.text, ent.label_) for ent in doc.ents])

nlp = spacy.load("en_ner_bc5cdr_md")
# https://allenai.github.io/scispacy/

def get_allen_conversations(df):
    allen_conversations = []

    for _, row in df.iterrows():
        speaker = row["Speaker"]
        text = row["Message"]
        if pd.isna(speaker) or pd.isna(text):
            continue
        if speaker == 'ALLEN':
          allen_conversations.append(text)

    return allen_conversations


allen_convos = get_allen_conversations(df)

def extract_entities(messages):
    entity_counter = Counter()
    for msg in messages:
        doc = nlp(msg)
        for ent in doc.ents:
            # We're only interested in CHEMICAL entities (e.g., supplements)
            if ent.label_ == "CHEMICAL":
                entity_counter[ent.text.lower()] += 1
    return entity_counter

supplement_counts = extract_entities(allen_convos)

print(supplement_counts)


