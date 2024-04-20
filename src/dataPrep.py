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

    df_data = df_all_data[['rating', 'text']].copy()
    df_data = df_data.rename(columns={'rating': 'stars', 'text': 'user_comment'})

    df_data['stars'] = pd.to_numeric(df_data['stars'], errors='coerce', downcast='integer')

    df_data = df_data.dropna(subset=['stars'])
    df_data['user_comment'] = df_data['user_comment'].astype(str)

    print("get_data -- End")

    df_data.to_csv(f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/processed/filtered_data_{state}.csv', mode='a', index=False)

    return 


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

    return


def retriveDataFromJsonForExploration(state):
    review_file_path = f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/review-{state}_10.json'
    df_review_data = pd.read_json(review_file_path, lines=True)

    metadata_file_path = f'/Users/aidankealey/Documents/fifth_year/CMPE_351/Project/CMPE351/data/meta-{state}.json'
    df_ratings = pd.read_json(metadata_file_path, lines=True)

    numRatings = df_ratings.shape[0]
    print(f'number of businesses: {numRatings}')
    df_ratings = df_ratings[df_ratings['category'].astype(str).str.contains('restaurant', case=False, na=False)]
    df_ratings = df_ratings.drop_duplicates(subset='gmap_id')
    numOfRestaurantRatings = df_ratings.shape[0]
    print(f'number of restaurants: {numOfRestaurantRatings}')

    df_all_data = pd.merge(df_review_data, df_ratings, on='gmap_id', how='inner')

    return df_all_data, numRatings, numOfRestaurantRatings

def exploreData(states):
    #stuff
    print("exploreData -- Start")

    dfs = []
    numRatings = 0
    numOfRestaurantRatings = 0

    for state in states:
        print(f'getting data for state: {state}')
        df_all_state_data, numRatings_state, numOfRestaurantRatings_state = retriveDataFromJsonForExploration(state)

        dfs.append(df_all_state_data[['rating', 'text', 'pics']].copy())
        numRatings += numRatings_state
        numOfRestaurantRatings += numOfRestaurantRatings_state
        print(f'got data for state: {state}')

    df_all_data = pd.concat(dfs, ignore_index=True)

    print(f'total number of ratings: {numRatings}')
    print(f'total number of restaurants ratings: {numOfRestaurantRatings}')

    # add pics stuff:
    picture_data_frame = pd.DataFrame(df_all_data, columns=['rating', 'pics'])
    picture_data_frame['pics'] = picture_data_frame['pics'].apply(lambda x: len(x) if x is not None else 0)
    picture_data_frame.rename(columns={'rating': 'Rating', 'pics': 'Num Pictures'}, inplace=True)
    
    recordsPics = len(picture_data_frame['Num Pictures'])
    print(f'number of records for Pics: {recordsPics}')
    pic_counts = picture_data_frame['Num Pictures'].value_counts().sort_index()
    zeroPics = pic_counts[0]
    print(f'number of reviews with no picture: {zeroPics}')
    onePlusPics = pic_counts[1:].sum()
    print(f'number of reviews with atleast one picture: {onePlusPics}')
    
    df_all_data = df_all_data.fillna('')
    recordsComment = len(df_all_data['text'])
    print(f'number commets (including NaN): {recordsComment}')
    df_all_data = df_all_data[df_all_data['text'] != '']
    recordsWithComment = len(df_all_data['text'])
    print(f'number of commets (without NaN): {recordsWithComment}')
    recordsWithoutComment = recordsComment - recordsWithComment
    print(f'number of NaN commets: {recordsWithoutComment}')

    df_data = df_all_data[['rating', 'text']].copy()
    df_data = df_data.rename(columns={'rating': 'stars', 'text': 'user_comment'})

    df_data['stars'] = pd.to_numeric(df_data['stars'], errors='coerce', downcast='integer')

    recordsStars = len(df_data['stars'])
    print(f'number stars (including NaN): {recordsStars}')

    df_data = df_data.dropna(subset=['stars'])
    df_data['user_comment'] = df_data['user_comment'].astype(str)

    recordsWithStars = len(df_data['stars'])
    print(f'number of stars (without NaN): {recordsWithStars}')
    recordsWithoutStars = recordsStars - recordsWithStars
    print(f'number of NaN stars: {recordsWithoutStars}')

    averageRating = df_data['stars'].mean()
    print(f'avergae rating in state: {state}, {averageRating:.2f} stars')
    star_counts = df_data['stars'].value_counts().sort_index()
    print(f'star distribution in state: {state} \n{star_counts}')

    print("exploreData -- End")

    return 



states = ['New_York', 'California', 'Texas']
# states = ['Montana']

# for s in states:
#     # get_data(s)
#     # updateCSV(s)

exploreData(states)