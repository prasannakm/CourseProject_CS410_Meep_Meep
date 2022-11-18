import pandas as pd

cities = ['Los Angeles', 'San Francisco', 'Chicago', 'Nashville', 'Boston', 'New Orleans', 'Austin', 'Denver']
state = ['CA', 'CA', 'IL', 'TN', 'MA', 'LA', 'TX', 'CO']
s = 0

# iterate through local file directory to retrieve the required columns for merging
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
    
    # remove non-ASCII characters
    u = df.select_dtypes(object)
    df[u.columns] = u.apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
    
    # remove html tags from the comments
    df['comments'] = df['comments'].str.replace(r'<[^<>]*>', '', regex=True)
    
    # take 10 random samples for each listing and drop any rows with duplicate comments
    df = df.groupby('id').sample(10, replace=True).drop_duplicates(['comments'])
        
    # rename host_location to location & from identified list variables, change the value to city, state
    df = df.rename(columns={'host_location':'location'})
    df['location'] = f'{city}, {state[s]}'
    s += 1
    
    # identify output directory and print to it
    filedir = f'{city}/{city}_Final_10.csv'
    df2 = df.to_csv(filedir, sep=',', index=False)
    

# merging the files
df_main = pd.DataFrame()

for city in cities:
    df3 = pd.read_csv(f'{city}/{city}_Final_10.csv', index_col=None)
    df_main = df_main.append(df3)

# print final dataset
df4 = df_main.to_csv('Data_Final_10.csv', sep=',', index=False)
