import pandas as pd
import nltk
from sklearn.model_selection import train_test_split

nltk.download('punkt')

def get_data(state):
    print("get_data -- Start")
    review_file_path = f'data/review-{state}_10.json'
    df_review_data = pd.read_json(review_file_path, lines=True)

    metadata_file_path = f'data/meta-{state}.json'
    df_ratings = pd.read_json(metadata_file_path, lines=True)

    df_ratings = df_ratings[df_ratings['category'].astype(str).str.contains('restaurant', case=False, na=False)]
    # df_ratings = df_ratings.drop_duplicates(subset='gmap_id')

    df_all_data = pd.merge(df_review_data, df_ratings, on='gmap_id', how='inner')

    df_bert_data = df_all_data[['gmap_id', 'name_y', 'rating', 'text']].copy()
    df_bert_data = df_bert_data.rename(columns={'name_y': 'business_name', 'rating': 'rating', 'text': 'user_comment'})

    print("get_data -- End")

    return df_bert_data


# custom tokenize to filter out missing or undefined data from the data files
def custom_tokenize(text):
    if type(text) != str:
        text = ''
    return nltk.word_tokenize(text)


def preprocess(df_data):
    print("preprocess -- Start")
    df_data = df_data.fillna('')

    # make all string lowercase
    df_data = df_data.map(lambda s: s.lower() if type(s) == str else s)

    # remove punctuation
    df_data = df_data.map(lambda s: s.replace(',', ' ') if type(s) == str else s)
    df_data = df_data.map(lambda s: s.replace('.', '') if type(s) == str else s)
    df_data = df_data.map(lambda s: s.replace('!', '') if type(s) == str else s)
    df_data = df_data.map(lambda s: s.replace('(', '') if type(s) == str else s)
    df_data = df_data.map(lambda s: s.replace(')', '') if type(s) == str else s)
    df_data = df_data.map(lambda s: s.replace('"', '') if type(s) == str else s)
    df_data = df_data.map(lambda s: s.replace('-', '') if type(s) == str else s)
    df_data = df_data.map(lambda s: s.replace(':', '') if type(s) == str else s)
    df_data = df_data.map(lambda s: s.replace(';', '') if type(s) == str else s)
    df_data = df_data.map(lambda s: s.replace('=', '') if type(s) == str else s)
    df_data = df_data.map(lambda s: s.replace('_', '') if type(s) == str else s)
    df_data = df_data.map(lambda s: s.replace('\'', '') if type(s) == str else s)

    # tokenize
    df_data['gmap_id'] = df_data['gmap_id'].apply(custom_tokenize)
    df_data['business_name'] = df_data['business_name'].apply(custom_tokenize)
    df_data['rating'] = df_data['rating'].apply(custom_tokenize)
    df_data['user_comment'] = df_data['user_comment'].apply(custom_tokenize)

    print("preprocess -- End")
    return df_data



def preprocess_data(df_Data):
    print("preprocess_data -- Start")
    df_data = preprocess(df_Data)

    df_train, df_test = train_test_split(df_data, test_size=0.1)  # train (train & val) -> 90%, test -> 10%
    df_train, df_val = train_test_split(df_train, test_size=0.2)  # train -> 80%, val -> 20%

    print("preprocess_data -- End")

    return df_train, df_val, df_test
