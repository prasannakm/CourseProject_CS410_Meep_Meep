import pandas as pd

cities = ['Los Angeles', 'San Francisco', 'Chicago', 'Nashville', 'Boston', 'New Orleans', 'Austin', 'Denver']
state = ['CA', 'CA', 'IL', 'TN', 'MA', 'LA', 'TX', 'CO']
s = 0
df_main = pd.DataFrame()

# iterate and read listings local file directory/keep the required columns for merging
for city in cities:
    listings = pd.read_csv(f'{city}/listings.csv')
    listings = listings.loc[:, ['id',
                         'name',
                         'host_location',
                         'neighbourhood_cleansed',
                         'latitude',
                         'longitude',
                         'price',
                         'review_scores_value']]
    
    # read reviews local file directory/keep the required columns for merging
    reviews = pd.read_csv(f'{city}/reviews.csv')
    reviews = reviews.loc[:, ['listing_id', 'comments']]

    # merge the listing and reviews datasets and keep only the required columns for our use case
    df = listings.merge(reviews, left_on='id', right_on='listing_id')
    df = df.loc[:, ['id',
                    'name',
                    'host_location',
                    'neighbourhood_cleansed',
                    'latitude',
                    'longitude',
                    'price',
                    'review_scores_value',
                    'comments']]
    
    # rename host_location to location & from identified list variables, change the value to city, state
    df = df.rename(columns={'host_location':'location'})
    df['location'] = f'{city}, {state[s]}'
    
    # remove non-ASCII characters
    u = df.select_dtypes(object)
    df[u.columns] = u.apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
    
    # remove html tags from the comments
    df['comments'] = df['comments'].str.replace(r'<[^<>]*>', '', regex=True)
    
    # remove punctuation
    df['comments'] = df['comments'].str.replace(r'[^\w\s]+', '', regex=True)
    
    # remove whitespace
    df['comments'] = df['comments'].str.strip()
                                                    
    # drop any rows with duplicate comments
    df = df.drop_duplicates(['comments'])
    
    # drop rows that are missing values
    df = df.dropna()
    
    # take 10 random samples for each listing
    df = df.groupby('id').sample(10, replace=True)
    
    # merging files
    df_main = df_main.append(df)
    
    # increment s to iterate through given city/state lists
    s += 1
    
# print final dataset
df1 = df_main.to_csv('Data_Final_10.csv', sep=',', index=False)
