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
        sentiment_score = compute_sentiment_score(probs)
        review_sentiments[aspect] = sentiment_score
        print(f"Sentiment score of aspect '{aspect}': {sentiment_score:.2f}")

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
with open('review_sentiments.csv', 'w', newline='') as csv_file:
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







# import torch
# import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# # Load models
# absa_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
# absa_model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
# sentiment_model_path = "nlptown/bert-base-multilingual-uncased-sentiment"
# sentiment_model = pipeline("sentiment-analysis", model=sentiment_model_path, tokenizer=sentiment_model_path)

# def get_aspect_sentiment(sentence, aspect):
#     inputs = absa_tokenizer(f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
#     outputs = absa_model(**inputs)
#     probs = F.softmax(outputs.logits, dim=1)
#     return probs.detach().numpy()[0]  # returns the probabilities for negative, neutral, positive

# def compute_scaled_sentiment(probs):
#     # Define the scale for each sentiment category
#     scale = {'negative': -100, 'neutral': 0, 'positive': 100}
#     # Calculate the weighted average sentiment
#     weighted_sentiment = sum(p * scale[s] for p, s in zip(probs, scale.keys()))
#     return weighted_sentiment

# def process_review(sentence, aspects):
#     print(f"\nReview: '{sentence}'")
#     review_sentiments = {}

#     for aspect in aspects:
#         probs = get_aspect_sentiment(sentence, aspect)
#         scaled_sentiment = compute_scaled_sentiment(probs)
#         review_sentiments[aspect] = scaled_sentiment
#         print(f"Scaled sentiment of aspect '{aspect}': {scaled_sentiment:.2f}")

#     return review_sentiments

# # Example data
# reviews = [
#     "We had a great experience at the restaurant, the food was delicious, but the service was kinda bad",
#     "Lovely place, although the wait time was too long, the staff were very friendly and the food was excellent",
#     "Terrible service, but the food was good enough, not the best place for a quick meal though"
# ]
# aspects = ["price", "service", "ambiance", "food"]

# # Process each review individually
# for review in reviews:
#     review_sentiments = process_review(review, aspects)
#     # for aspect, sentiment in review_sentiments.items():
#     #     print(f"Final scaled sentiment for '{aspect}' in this review: {sentiment:.2f}")
