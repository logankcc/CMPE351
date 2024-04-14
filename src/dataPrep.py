import pandas as pd

def get_data(state):
    print("get_data -- Start")
    review_file_path = f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/review-{state}_10.json'
    df_review_data = pd.read_json(review_file_path, lines=True)

    metadata_file_path = f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/meta-{state}.json'
    df_ratings = pd.read_json(metadata_file_path, lines=True)

    df_ratings = df_ratings[df_ratings['category'].astype(str).str.contains('restaurant', case=False, na=False)]
    df_ratings = df_ratings.drop_duplicates(subset='gmap_id')

    df_all_data = pd.merge(df_review_data, df_ratings, on='gmap_id', how='inner')

    df_all_data = df_all_data.fillna('')
    df_all_data = df_all_data[df_all_data['text'] != '']  # drop reviews with no comments

    df_bert_data = df_all_data[['rating', 'text']].copy()
    df_bert_data = df_bert_data.rename(columns={'rating': 'stars', 'text': 'user_comment'})

    df_bert_data['region'] = f'{state}'

    print("get_data -- End")

    return df_bert_data