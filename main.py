import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer 
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


movies = pd.read_csv('C:\\Users\\goyal\\Desktop\\project python\\movie_recomendation\\tmdb_5000_movies.csv')
credits = pd.read_csv('C:\\Users\\goyal\\Desktop\\project python\\movie_recomendation\\tmdb_5000_credits.csv')

movies = movies.merge(credits,on='title')

# required column : genres , id , keywords,title,overview,cast,crew
movies = movies[['movie_id', 'title', 'overview', 'keywords', 'cast', 'crew','genres']]


# removing missing data
movies.isnull().sum()
movies.duplicated().sum()
movies.dropna(inplace=True)

def convert(obj):
    l = []
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l

movies['genres'] = movies['genres'].apply(convert)

movies['keywords'] = movies['keywords'].apply(convert)
def convert3(obj):
    l = []
    counter =0
    for i in ast.literal_eval(obj):
        if counter !=3:
            l.append(i['name'])
            counter +=1
    return l
movies['cast'] = movies['cast'].apply(convert3)

def fetch_director (obj):
    l =[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            l.append(i['name'])
            break
    return l

movies['crew'] = movies['crew'].apply(fetch_director) 
movies['overview']=movies['overview'].apply(lambda x:x.split())   

movies['genres']=movies['genres'].apply(lambda x:[i .replace (" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies ['crew'] = movies['crew'].apply(lambda x:[i.replace (" ","")for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace (" ","")for i in x])

movies['tags'] = movies['cast']+movies['crew']+movies['overview']+movies['genres']+movies['keywords']

new_df = movies[['movie_id','title','tags']]

new_df['tags']=  new_df['tags'].apply(lambda x:" ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)    

new_df['tags'] = new_df['tags'].apply(stem)


cv = CountVectorizer(max_features= 5000,stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray() # type: ignore



similarity = cosine_similarity(vectors)
def recommend(movie):
    movie_index = new_df[new_df['title']== movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse = True ,key=lambda x:x[1])[1:6]
    for i in movies_list:
        st.write(new_df.iloc[i[0]].title)
    



st.title('Movie Recommender System')
option = st.selectbox( 'select the movie',new_df['title'])

if st.button('Search'):
    st.write(recommend(option))


