import numpy as np
import torch
import torch.nn.functional as F
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import dataPrep
import pandas as pd
from datetime import datetime

# Load models
absa_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
absa_model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
sentiment_model_path = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_model = pipeline("sentiment-analysis", model=sentiment_model_path, tokenizer=sentiment_model_path)

def get_aspect_sentiment(sentence, aspect):
    inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
    outputs = absa_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    return probs.detach().numpy()[0]  # returns the probabilities for negative, neutral, positive

def compute_sentiment_score(probs):
    # Sentiment weights map
    if (probs[0] < probs[2]):
        output = -1 * probs[0] + -1 * probs[1] + 1 * probs[2]
    elif (probs[0] > probs[2]):
        output = -1 * probs[0] + 1 * probs[1] + 1 * probs[2]
    else:
        output = -1 * probs[0] + 0 * probs[1] + 1 * probs[2]

    scaled_score = 100 * output
    return scaled_score

def process_review(sentence, aspects):
    review_sentiments = {}

    for aspect in aspects:
        probs = get_aspect_sentiment(sentence, aspect)
        if np.max(probs) > 0.6:
            sentiment_score = compute_sentiment_score(probs)
            review_sentiments[aspect] = sentiment_score
        else:
            sentiment_score = None
            review_sentiments[aspect] = sentiment_score

    return review_sentiments

def process_row(row, aspects):
    review_sentiments = process_review(row['user_comment'], aspects)
    for aspect, sentiment_score in review_sentiments.items():
        row[aspect] = sentiment_score
    return row

states = ['New_York', 'California', 'Texas']
# states = ['Montana']
aspects = ["price", "service", "ambiance", "food"]
ratingCSV = ["stars", "price", "service", "ambiance", "food"]
regionCSV = ["region", "price", "service", "ambiance", "food"]

for state in states:
    print(f'Running {state}')
    df_data = dataPrep.get_data(state=state)
    df_data = df_data.head(100)

    for column in aspects:
        df_data[column] = None

    print(df_data.shape)

    df_data = df_data.apply(process_row, aspects=aspects, axis=1)

    # startTime = datetime.now()

    # for idx, record in df_data.iterrows():
    #     if (idx % 10000 == 0):
    #         time = str(datetime.now() - startTime).split('.')[0]
    #         print(f'Index: {idx} -- Time from start: {time}')

    #     review_sentiments = process_review(record['user_comment'], aspects)

    #     for aspect, sentiment_score in review_sentiments.items():
    #         df_data.at[idx, aspect] = sentiment_score

    print(df_data.head())

    df_data[ratingCSV].to_csv(f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/review_sentiments_{state}_stars.csv', index=False)
    df_data[regionCSV].to_csv(f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/review_sentiments_{state}_geo.csv', index=False)
    
