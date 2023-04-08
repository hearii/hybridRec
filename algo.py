#%matplotlib inline
import pandas as pd
import numpy as np
import pickle
import difflib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
#from surprise import Dataset
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD
from configs import working_dir
from mongodb import get_database

import warnings; warnings.simplefilter('ignore')
dbname = get_database()
collection_name1 = dbname["movie_metadata"]
movies_metadata=(list(collection_name1.find()))

collection_name2 = dbname["ratings_small"]
ratings_small_db=(list(collection_name2.find()))

collection_name3= dbname["link_small"]
links_small_db=(list(collection_name3.find()))

#adding data to main dataframe (md)

md = pd.DataFrame(movies_metadata)
md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()
m = vote_counts.quantile(0.95)
md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)


#adding data to small main datafram (smd)

links_small = pd.DataFrame(links_small_db)
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
md = md.drop([19730, 29503, 35587])                                                    #data not present in these cells
md['id'] = md['id'].astype('int')
smd = md[md['id'].isin(links_small)]
smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])
md['id'] = md['id'].astype('int')
smd = md[md['id'].isin(links_small)]


def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan


id_map = pd.DataFrame(links_small_db)[['movieId','tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
indices_map = id_map.set_index('id')
#print(id_map.head())

reader = Reader()
ratings = pd.DataFrame(ratings_small_db)
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
svd = SVD()
trainset = data.build_full_trainset()
svd.fit(trainset)


#Rec fxn


def hybrid(userId, titles):
    
    idx = indices[titles]
    tmdbId = id_map.loc[titles]['id']
    #print(idx)
    movie_id = id_map.loc[titles]['movieId']
    
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    movies_recommended=movies.head(10).to_dict(orient='records')
    return movies_recommended

