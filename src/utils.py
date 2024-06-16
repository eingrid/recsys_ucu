import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score


def load_data(data_path):
    """
    Load data from the specified data_path.

    Args:
        data_path (str): The path to the root data directory.

    Returns:
        tuple: A tuple containing three pandas DataFrames: users, ratings, and movies.
    """
    unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
    users = pd.read_table(os.path.join(data_path,'users.dat'), sep='::', header=None, names=unames, engine='python')

    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_table(os.path.join(data_path,'ratings.dat'), sep='::',header=None, names=rnames, engine='python')

    mnames = ['movie_id', 'title', 'genres']
    movies = pd.read_table(os.path.join(data_path,'movies.dat'), sep='::',header=None, names=mnames, engine='python',encoding='latin-1')
    return users, ratings, movies

def evaluate_model(test_ratings, user_item_matrix, similarity_df, get_recommendations, n_recommendations=10):
    precisions = []
    recalls = []
    f1s = []

    for user_id in test_ratings['UserID'].unique():
        if user_id not in user_item_matrix.index:
            continue
        true_items = test_ratings[test_ratings['UserID'] == user_id]['MovieID'].tolist()
        if len(true_items) == 0:
            continue
        recommended_items = get_recommendations(user_id, user_item_matrix, similarity_df, n_recommendations)
        if len(recommended_items) == 0:
            continue
        true_binary = [1 if item in true_items else 0 for item in recommended_items]
        recommended_binary = [1] * len(recommended_items)  # All recommended items are considered 1
        if sum(true_binary) == 0:
            continue
        precision = precision_score(true_binary, recommended_binary)
        recall = recall_score(true_binary, recommended_binary)
        f1 = f1_score(true_binary, recommended_binary)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    average_precision = round(sum(precisions) / len(precisions), 2) if precisions else 0
    average_recall = round(sum(recalls) / len(recalls), 2) if recalls else 0
    average_f1 = round(sum(f1s) / len(f1s), 2) if f1s else 0

    return average_precision, average_recall, average_f1


def load_data(movies_path, ratings_path):
    movies = pd.read_csv(movies_path, sep='::', header=None, names=['MovieID', 'Title', 'Genres'], engine='python', encoding='latin-1')
    ratings = pd.read_csv(ratings_path, sep='::', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'], engine='python', encoding='latin-1')
    return movies, ratings
