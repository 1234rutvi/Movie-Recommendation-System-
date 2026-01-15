# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 16:39:54 2025

@author: HP
"""
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request
import requests

# ----------------- LOAD AND PROCESS DATA -----------------
movies = pd.read_csv(r"C:\Users\HP\Downloads\tmdb_5000_movies.csv")
credits = pd.read_csv(r"C:\Users\HP\Downloads\tmdb_5000_credits.csv")
movies = movies.merge(credits, on="title")
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

def convert(text):
    return [i['name'] for i in ast.literal_eval(text)]

def convert_cast(text):
    L, counter = [], 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
            counter += 1
    return L

def fetch_director(text):
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            return [i['name']]
    return []

def clean_data(text):
    return [i.replace(" ", "").lower() for i in text]

movies['genres'] = movies['genres'].apply(convert).apply(clean_data)
movies['keywords'] = movies['keywords'].apply(convert).apply(clean_data)
movies['cast'] = movies['cast'].apply(convert_cast).apply(clean_data)
movies['crew'] = movies['crew'].apply(fetch_director).apply(clean_data)
movies['overview'] = movies['overview'].apply(lambda x: x.lower().split())

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

# ----------------- FLASK SETUP -----------------
app = Flask(__name__)
API_KEY = "b25a5793d33d4164e54869a6b9c8df22"  # Replace with your TMDB key

def get_movie_with_api_key(movie_id, api_key):
    """Fetch movie details using API key in query parameter"""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {
        'api_key': api_key,
        'language': 'en-US'
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.text} 
    
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    try:
        data = requests.get(url, timeout=5).json()

        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path

    except Exception as e:
        print("Poster fetch failed:", e)

    # ✅ fallback image
    return "https://via.placeholder.com/200x300?text=No+Poster"


def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    names = []
    posters = []

    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        names.append(movies.iloc[i[0]].title)
        posters.append(fetch_poster(movie_id))

    return names, posters



# ----------------- FLASK ROUTE -----------------
@app.route('/', methods=['GET', 'POST'])
def home():
    movie_names = movies['title'].values
    recommendations = []

    if request.method == 'POST':
        movie = request.form.get('movie')
        names, posters = recommend(movie)
        recommendations = zip(names, posters)

    return render_template('index.html', movie_names=movie_names, recommendations=recommendations)

# ----------------- RUN SERVER -----------------
if __name__ == "__main__":
    app.run(debug=True)










