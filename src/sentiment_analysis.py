import numpy as np
import torch.nn.functional as F
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

    # scaled_score = 100 * output
    return 100 * output

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

    print(df_data.shape)

    batch_size = 2500
    for i in range(0, len(df_data), batch_size):
        startTime = datetime.now()
        print(f'Batch: {i} -- start time: {startTime.strftime("%H:%M:%S")}')
        batch_df = df_data.iloc[i:i+batch_size]
        for column in aspects:
            batch_df[column] = None
        batch_df = batch_df.apply(process_row, aspects=aspects, axis=1)
        batch_df[ratingCSV].to_csv(f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/review_sentiments_{state}_stars.csv', mode='a', index=False, header=not i)
        batch_df['region'] = f'{state}'
        batch_df[regionCSV].to_csv(f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/review_sentiments_{state}_geo.csv', mode='a', index=False, header=not i)
        endTime = datetime.now()
        deltaTime = endTime - startTime
        print(f'end time: {endTime.strftime("%H:%M:%S")} \ndeltaTime: {deltaTime.strftime("%H:%M:%S")}')


    # df_data = df_data.apply(process_row, aspects=aspects, axis=1)

    # df_data['region'] = f'{state}'

    # print(df_data.head())

    # df_data[ratingCSV].to_csv(f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/review_sentiments_{state}_stars.csv', index=False)
    # df_data[regionCSV].to_csv(f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/review_sentiments_{state}_geo.csv', index=False)
    
