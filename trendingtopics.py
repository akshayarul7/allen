
import pandas as pd
from collections import Counter, defaultdict
import spacy




df = pd.read_excel("conversations1.xlsx")

column_dictionary = {'Ways2Well - 05/20/2025 00:00 to 05/20/2025 23:59': 'Session ID', "Unnamed: 1": "Avatar Id", "Unnamed: 2": "Log #", "Unnamed: 3": "Date", "Unnamed: 4": "Avatar Title", "Unnamed: 5": "Talk Time", "Unnamed: 6":"Speaker", "Unnamed: 7": "Message"}
df.rename(columns = column_dictionary, inplace= True)



def get_user_conversations(df):
    user_conversations = []

    for _, row in df.iterrows():
        speaker = row["Speaker"]
        text = row["Message"]
        if pd.isna(speaker) or pd.isna(text):
            continue
        if speaker == 'Guest User':
          user_conversations.append(text)

    return user_conversations


user = get_user_conversations(df)
print(user)
