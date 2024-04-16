import numpy as np
import torch
import torch.nn.functional as F
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# resultsData = []  

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
    sentiment_weights = {'negative': -1, 'neutral': 0, 'positive': 1}
    # Calculate the sentiment score as a weighted sum
    sentiment_score = sum(weight * prob for weight, prob in zip(sentiment_weights.values(), probs))
    # Optionally scale the result to a different range, e.g., -100 to 100
    scaled_score = 100 * sentiment_score
    return scaled_score

def process_review(sentence, aspects):
    print(f"\nReview: '{sentence}'")
    review_sentiments = {}

    for aspect in aspects:
        probs = get_aspect_sentiment(sentence, aspect)
        # need to drop attributes that do not have a sentiment confidence score above 70 
        print(probs)
        if np.max(probs) > 0.6:
            sentiment_score = compute_sentiment_score(probs)
            review_sentiments[aspect] = sentiment_score
            print(f"Sentiment score of aspect '{aspect}': {sentiment_score:.2f}")
        else:
            sentiment_score = None
            review_sentiments[aspect] = sentiment_score
            print(f"Confidence of sentiment score of aspect '{aspect}' is to low, do not consider")

    return review_sentiments

# Example data
reviews = [
    "We had a great experience at the restaurant, the food was delicious, but the service was kinda bad",
    "Lovely place, although the wait time was too long, the staff were very friendly and the food was excellent",
    "Terrible service, but the food was good enough, not the best place for a quick meal though", 
    "It was a nice place, but the food was only okay", 
    "It was a nice place, but the food was okay", 
    "It was a nice place, but the food was horrible"
]
aspects = ["price", "service", "ambiance", "food"]

fieldnames = ['star_rating'] + aspects

# Open CSV file in write mode and write headers
with open('data/review_sentiments.csv', 'w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Write data for each review
    for review in reviews:
        review_sentiments = process_review(review, aspects)
        # star_rating = review['star_rating']

        # Create a dictionary for this review
        # review_data = {'star_rating': star_rating}
        review_data = {}

        # Add aspect scores to the review data
        for aspect, sentiment_score in review_sentiments.items():
            review_data[aspect] = sentiment_score

        # Write review data to CSV file
        writer.writerow(review_data)

