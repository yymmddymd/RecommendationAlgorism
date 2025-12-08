import pandas as pd
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

app = Flask(__name__)

def load_data():
    try:
        movies = pd.read_csv('movies_100k.csv', sep='|', encoding='latin-1', 
                             header=None, usecols=[0, 1], names=['movieId', 'title'])
        movies['movieId'] = pd.to_numeric(movies['movieId'], errors='coerce').astype('Int64')
        movies.dropna(subset=['movieId'], inplace=True)

        ratings = pd.read_csv('ratings_100k.csv')
        ratings['movieId'] = pd.to_numeric(ratings['movieId'], errors='coerce').astype('Int64')
        
        return movies, ratings
    except FileNotFoundError:
        print("エラー: 必要なCSVファイルが見つかりません。")
        exit()

movies_df, ratings_df = load_data()
df_merged = ratings_df.merge(movies_df, on='movieId')
user_movie_matrix = df_merged.pivot_table(index='movieId', columns='userId', values='rating').fillna(0)
sparse_matrix = csr_matrix(user_movie_matrix.values)
item_similarity = cosine_similarity(sparse_matrix)
item_similarity_df = pd.DataFrame(item_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)
title_to_id = movies_df.set_index('title')['movieId'].to_dict()

def get_recommendations(selected_titles):
    """選択された映画に基づいておすすめ映画を返す"""
    if not selected_titles:
        return []

    total_scores = pd.Series(0.0, index=item_similarity_df.index)

    valid_selected_ids = []
    for title in selected_titles:
        movie_id = title_to_id.get(title)
        if movie_id in item_similarity_df.index:
            total_scores += item_similarity_df.loc[movie_id]
            valid_selected_ids.append(movie_id)

    if valid_selected_ids:
        total_scores.loc[valid_selected_ids] = -1.0

    top_ids = total_scores.sort_values(ascending=False).head(5).index
    recommended_titles = movies_df[movies_df['movieId'].isin(top_ids)]['title'].tolist()
    return recommended_titles

@app.route('/', methods=['GET'])
def index():
    titles = sorted(movies_df['title'].unique().tolist())
    return render_template('index.html', movie_titles=titles)

@app.route('/recommend', methods=['POST'])
def recommend():
    selected_movies = request.form.getlist('selected_movies')
    selected_movies = [m for m in selected_movies if m]
    
    recommendations = get_recommendations(selected_movies)
    return render_template('recommendations.html', 
                           selected_movies=selected_movies, 
                           recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)