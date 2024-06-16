import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import matplotlib.pyplot as plt


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


def convert_keys_to_int(d):
    if isinstance(d, dict):
        new_dict = {}
        for k, v in d.items():
            new_key = int(k) if k.isdigit() else k
            new_dict[new_key] = convert_keys_to_int(v)
        return new_dict
    elif isinstance(d, list):
        return [convert_keys_to_int(i) for i in d]
    else:
        return d

def load_baseline_rec_result(file_path='../../artifacts/results.json'):
    with open(file_path, 'r') as file:
        baseline_rec_result = json.load(file)
    return convert_keys_to_int(baseline_rec_result)
# Function to plot the metrics on a grid
def plot_metrics_grid(results, metrics):
    num_metrics = len(metrics)
    num_cols = 2  # Number of columns in the grid
    num_rows = (num_metrics + num_cols - 1) // num_cols  # Calculate rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    axes = axes.flatten()  # Flatten to easily iterate over them

    for idx, metric_name in enumerate(metrics):
        ax = axes[idx]
        for recommender in results:
            ks = sorted(results[recommender].keys())  # Sorting keys
            metric_values = [results[recommender][k][metric_name] for k in ks]
            ax.plot(ks, metric_values, label=recommender)

        ax.set_xlabel('k')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} vs k')
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots
    for idx in range(len(metrics), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()

