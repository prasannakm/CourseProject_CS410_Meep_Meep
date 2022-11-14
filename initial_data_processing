import pandas as pd

cities = ['Los Angeles', 'San Francisco', 'Chicago', 'Nashville', 'Boston', 'New Orleans', 'Austin', 'Denver']
state = ['CA', 'CA', 'IL', 'TN', 'MA', 'LA', 'TX', 'CO']
s = 0

# iterate through local file directory to retrieve the required columns for merging
for city in cities:
    print(city)
    listings = pd.read_csv(f'{city}/listings.csv')
    listings = listings.loc[:, ['id',
                         'name',
                         'host_location',
                         'neighbourhood_cleansed',
                         'latitude',
                         'longitude',
                         'price',
                         'review_scores_value']]
    
    reviews = pd.read_csv(f'{city}/reviews.csv')
    reviews = reviews.loc[:, ['listing_id', 'comments']]

    # merge the listing and reviews datasets and retrieve only the required columns for our use case
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
    
    # take the top 10 of each listing from the dataset
    df = df.groupby('id').head(10)
    
    # for 10 random samples
    # df = df.groupby('id').sample(10, replace=True)
    
    # rename host_location to location & from identified list variables, change the value to city, state
     df = df.rename(columns={'host_location':'location'})
    df['location'] = f'{city}, {state[s]}'
    s += 1
    
    # identify output directory and print to it
    filedir = f'{city}/{city}_Final_10.csv'
    df2 = df.to_csv(filedir, sep=',')
