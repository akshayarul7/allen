from transformers import pipeline
import pandas as pd



# yo added this comment 
df = pd.read_excel("conversations.xlsx")  
df.rename(columns={"Unnamed: 6":"Speaker", "Unnamed: 7": "Message"}, inplace=True)

sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

labels = {
    "LABEL_0":"Negative",
    "LABEL_1":"Netural",
    "LABEL_2":"Positive"
}

# filter customer messages
customer_msgs = df[df["Speaker"] == "Guest User"]

# finding label (negative, neutral, positive)
df.loc[customer_msgs.index, "Sentiment"] = customer_msgs["Message"].apply(
    lambda x: labels[sentiment_pipeline(str(x))[0]['label']]
)

# finding confidence score
df.loc[customer_msgs.index, "Score"] = customer_msgs["Message"].apply(
    lambda x: sentiment_pipeline(str(x))[0]['score']
)

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
print(df[["Message", "Sentiment", "Score"]].dropna().head(50))