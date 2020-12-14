from math import sqrt
# from baselines import MeanofMeans, GlobalMean
# from surprise import AlgoBase
import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, url_for, request, jsonify



app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('legacy.html')

@app.route('/next_gen', methods=['GET'])
def next_gen():
    return render_template('next_gen.html')

@app.route('/legacy', methods=['POST'])
def legacy_recommend():
    user_data = request.json
    user_id, movie_id, number_movies = user_data['user_id'], user_data['movie_id'], user_data['number_movies']
    movie_list = _recommend_movie(user_id, movie_id, number_movies)
    return jsonify({'result': movie_list})

def _recommend_movie(user_id, movie_id, number_movies):
    """Generate recommended movies as a list of list. Each pair follows the pattern [movie_id, movie_name]

    Args:
        user_id (int): user id
        movie_id (int): movie_id that the user watched
        number_movies (int): number of movies to return
    Returns:
        a list of list. Each pair follows the pattern [movie_id, movie_name]
    """
    from baselines import MeanofMeans, GlobalMean
    all_recs = []
    user = user_id
    with open('model.sav','rb') as f:
        model = pickle.load(f)
    movie_df = pd.read_csv('movies.csv')
    for movie in range(len(model.item_means)):
        all_recs.append(model.estimate(user,movie))
    indices = np.argsort(all_recs)[:-number_movies-1:-1]
    full_recs = []
    for idx,movie in enumerate(indices):
        rec = []
        rec.append(movie) 
        rec.append(movie_df['title'].iloc[movie])
        full_recs.append(rec)
    for i in range(len(indices)):
        full_recs[i][0] = int(full_recs[i][0])
    return full_recs

@app.route('/next_gen', methods=['POST'])
def next_gen_recommend():
    user_data = request.json
    user_id, movie_id, number_movies = user_data['user_id'], user_data['movie_id'], user_data['number_movies']
    movie_list = _recommend_movie_next_gen(user_id, movie_id, number_movies)
    return jsonify({'result': movie_list})

def _recommend_movie_next_gen(user_id, movie_id, number_movies):
    """Generate recommended movies as a list of list. Each pair follows the pattern [movie_id, movie_name]

    Args:
        user_id (int): user id
        movie_id (int): movie_id that the user watched
        number_movies (int): number of movies to return
    Returns:
        a list of list. Each pair follows the pattern [movie_id, movie_name]
    """
    from surprise.prediction_algorithms.matrix_factorization import SVDpp
    all_recs = []
    user = user_id
    with open('svdpp.sav','rb') as f:
        model = pickle.load(f)
    movie_df = pd.read_csv('movies.csv')
    for movie in range(len(model.qi)):
        all_recs.append(model.predict(user,movie).est)
    indices = np.argsort(all_recs)[:-number_movies-1:-1]
    full_recs = []
    for idx,movie in enumerate(indices):
        rec = []
        rec.append(movie) 
        rec.append(movie_df['title'].iloc[movie])
        full_recs.append(rec)
    for i in range(len(indices)):
        full_recs[i][0] = int(full_recs[i][0])
    return full_recs


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
# model = MeanofMeans()
    
# app.run(host='0.0.0.0', port=8000, debug=False)