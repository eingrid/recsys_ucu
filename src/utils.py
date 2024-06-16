import pandas as pd
import os

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