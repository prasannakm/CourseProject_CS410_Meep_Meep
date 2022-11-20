import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
nltk.download('vader_lexicon')
nltk.download('punkt')

from nltk.sentiment.vader import SentimentIntensityAnalyzer as vad

from sklearn import feature_extraction
from sklearn.metrics import confusion_matrix
from sklearn import model_selection as ms
from sklearn import naive_bayes
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")


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
    
    # rename columns & from identified list variables, change the value to city, state
    df = df.rename(columns={'host_location':'location', 'review_scores_value':'review_scores'})
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
    
    # drop any rows with duplicate comments
    df = df.drop_duplicates(['comments'])
    
    # drop rows that are missing values
    df = df.dropna()
    
    # convert the comment values to string and lowercase
    df['comments'] = df['comments'].astype(str)
    df['comments'] = df['comments'].str.lower()

    # tokenize the reviews to word token
    df['comments'] = df['comments'].apply(word_tokenize)

    # remove stop words from the tokens
    df['comments'] = df['comments'].apply(lambda x: ' '.join([item for item in x if item not in stop]))
    
    # merging files
    df_main = df_main.append(df)
    
    # increment s to iterate through given city/state lists
    s += 1
    
# print final dataset
#df1 = df_main.to_csv('Dataset_Final.csv', sep=',', index=False)

# Vader Sentiment Analysis
# making additional columns for sentiment score in the vader dataframe
sentiment = vad()

sen = ['Positive', 'Negative', 'Neutral']
sentiments = [sentiment.polarity_scores(i) for i in df_main['comments'].values]
df_main['Vad_Negative Score'] = [i['neg'] for i in sentiments]
df_main['Vad_Positive Score'] = [i['pos'] for i in sentiments]
df_main['Vad_Neutral Score'] = [i['neu'] for i in sentiments]
df_main['Vad_Compound Score'] = [i['compound'] for i in sentiments]
score = df_main['Vad_Compound Score'].values
t = []

for i in score:
    if i >= 0.05:
        t.append('Positive')
    elif i <= -0.05:
        t.append('Negative')
    else:
        t.append('Neutral')
        
df_main['Vad_Overall_Sentiment'] = t

# Vader Visualizations
'''
sns.countplot(df_main['Vad_Overall_Sentiment'])
explode = [0, 0.1, 0.1]
df_main["Vad_Overall_Sentiment"].value_counts().plot.pie(title="Over All Sentiment",autopct='%1.1f%%', 
                        explode = explode
                                 )
time_df = df_main.groupby(['Vad_Overall_Sentiment','location'])['comments'].count().reset_index()
plt.figure(figsize=(30, 8))
sns.barplot(data=time_df, x='location',y='comments', hue='Vad_Overall_Sentiment')
plt.show()

time_df = df_main.groupby(['Vad_Overall_Sentiment','review_scores'])['comments'].count().reset_index()
plt.figure(figsize=(30,8))
sns.barplot(data=time_df, x='review_scores', y='comments', hue='Vad_Overall_Sentiment')
plt.show()
'''
