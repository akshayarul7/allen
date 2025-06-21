import pandas as pd
from collections import defaultdict


df = pd.read_excel("conversations.xlsx")  

column_dictionary = {'Ways2Well - 05/20/2025 00:00 to 05/20/2025 23:59': 'Session ID', "Unnamed: 1": "Avatar Id", "Unnamed: 2": "Log #", "Unnamed: 3": "Date", "Unnamed: 4": "Avatar Title", "Unnamed: 5": "Talk Time", "Unnamed: 6":"Speaker", "Unnamed: 7": "Message"}
df.rename(columns = column_dictionary, inplace= True)



def get_conversations(df):
  conv_count = defaultdict(int)  
  conversations = defaultdict(list)  
  for _, row in df.iterrows():
      session_id = row["Session ID"]
      log_num = row["Log #"]
      speaker = row["Speaker"]
      text = row["Message"]
      if pd.isna(session_id) or pd.isna(speaker) or pd.isna(text):
        continue

      # log_num == 0 is the start of a new conversation for this session
      if log_num == 0:
          conv_count[session_id] += 1

      # the key is the current session_id. If there are multiple conversations with same session_id, keep track of that as well
      key = (session_id, conv_count[session_id])

      # add message to the key's value as a dictionary
      conversations[key].append({"Speaker": speaker, "Message": text})


conversations = get_conversations(df)