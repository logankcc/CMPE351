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

    df_bert_data['stars'] = pd.to_numeric(df_bert_data['stars'], errors='coerce', downcast='integer')

    df_bert_data = df_bert_data.dropna(subset=['stars'])
    df_bert_data['user_comment'] = df_bert_data['user_comment'].astype(str)


    print("get_data -- End")

    df_bert_data.to_csv(f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/processed/filtered_data_{state}.csv', mode='a', index=False)


    return df_bert_data


def updateCSV(state):
    review_file_path = f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/new/review_sentiments_{state}_stars.csv'
    file_star = pd.read_csv(review_file_path, header=None)

    file_star = file_star.drop(file_star.columns[1], axis=1)
    file_star.to_csv(f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/review_sentiments_stars.csv', mode='a', index=False, header=False)

    review_file_path2 = f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/new/review_sentiments_{state}_geo.csv'
    file_geo = pd.read_csv(review_file_path2, header=None)

    last_column = file_geo.iloc[:, -1]
    file_geo = file_geo.drop(file_geo.columns[0], axis=1)
    file_geo = file_geo.drop(file_geo.columns[0], axis=1)
    file_geo = file_geo.drop(file_geo.columns[-1], axis=1)
    file_geo.insert(0, 'state', last_column)

    file_geo.to_csv(f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/review_sentiments_geo.csv', mode='a', index=False, header=False)


states = ['New_York', 'California', 'Texas']

# for s in states:
#     # get_data(s)
#     updateCSV(s)