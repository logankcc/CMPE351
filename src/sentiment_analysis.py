import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
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
    probs = probs
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


# def process_review_sentiments(row):
#     review_sentiments = process_review(row['user_comment'], aspects)
#     for aspect, sentiment_score in review_sentiments.items():
#         row[aspect] = sentiment_score
#     return row

def process_review_sentiments(chunk):
    for idx, row in chunk.iterrows():
        review_sentiments = process_review(row['user_comment'], aspects)
        for aspect, sentiment_score in review_sentiments.items():
            chunk.at[idx, aspect] = sentiment_score
    return chunk


# states = ['New_York', 'California', 'Texas']
# state = 'Montana'
state = 'New_York'
# state = 'California'
# state = 'Texas'
aspects = ["price", "service", "ambiance", "food"]
ratingCSV = ["stars", "price", "service", "ambiance", "food"]
regionCSV = ["region", "price", "service", "ambiance", "food"]

# for state in states:
print(f'Running {state}')
# df_data = dataPrep.get_data(state=state)

review_file_path = f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/processed/filtered_data_{state}.csv'
# my_data = pd.read_csv(review_file_path)

chunks = pd.read_csv(review_file_path, chunksize=5000)

# Process each chunk and write to CSV files
for chunk in chunks:
    startTime = datetime.now()
    print(f'start -- time: {startTime.strftime("%H:%M:%S")}')
    processed_chunk = process_review_sentiments(chunk)
    processed_chunk.to_csv(f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/review_sentiments_{state}_stars.csv', mode='a', index=False, header=False)
    processed_chunk['region'] = state
    processed_chunk.to_csv(f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/review_sentiments_{state}_geo.csv', mode='a', index=False, header=False)
    endTime = datetime.now()
    deltaTime = endTime - startTime
    print(f'end of state\nend time: {endTime.strftime("%H:%M:%S")} \ndeltaTime: {deltaTime}')


# print("get_data -- End")

# print(df_data)

# i = 1

# for df_batch in df_data:
#     startTime = datetime.now()
#     print(df_batch)
#     print(f'Batch: {i} -- start time: {startTime.strftime("%H:%M:%S")}')
#     for column in aspects:
#         df_batch[column] = None

#     for idx, record in df_batch.iterrows():
#         review_sentiments = process_review(record['user_comment'], aspects)

#         for aspect, sentiment_score in review_sentiments.items():
#             df_batch.at[idx, aspect] = sentiment_score
    
#     df_batch[ratingCSV].to_csv(f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/review_sentiments_{state}_stars.csv', mode='a', index=False, header=not i)
#     df_batch['region'] = f'{state}'
#     df_batch[regionCSV].to_csv(f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/review_sentiments_{state}_geo.csv', mode='a', index=False, header=not i)
#     endTime = datetime.now()
#     deltaTime = endTime - startTime
#     print(f'end time: {endTime.strftime("%H:%M:%S")} \ndeltaTime: {deltaTime}')
#     i += 1

# batch_size = 100
# for i in range(0, len(df_data), batch_size):
#     startTime = datetime.now()
#     print(f'Batch: {i} -- start time: {startTime.strftime("%H:%M:%S")}')
#     df_batch = df_data[i:i+batch_size]
#     for column in aspects:
#         df_batch[column] = None

#     for idx, record in df_batch.iterrows():
#         review_sentiments = process_review(record['user_comment'], aspects)

#         for aspect, sentiment_score in review_sentiments.items():
#             df_batch.at[idx, aspect] = sentiment_score
    
#     df_batch[ratingCSV].to_csv(f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/review_sentiments_{state}_stars.csv', mode='a', index=False, header=not i)
#     df_batch['region'] = f'{state}'
#     df_batch[regionCSV].to_csv(f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/review_sentiments_{state}_geo.csv', mode='a', index=False, header=not i)
#     endTime = datetime.now()
#     deltaTime = endTime - startTime
#     print(f'end time: {endTime.strftime("%H:%M:%S")} \ndeltaTime: {deltaTime}')


# df_data = df_data.apply(process_row, aspects=aspects, axis=1)

# df_data['region'] = f'{state}'

# print(df_data.head())

# df_data[ratingCSV].to_csv(f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/review_sentiments_{state}_stars.csv', index=False)
# df_data[regionCSV].to_csv(f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/review_sentiments_{state}_geo.csv', index=False)

