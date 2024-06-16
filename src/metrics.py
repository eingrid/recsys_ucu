import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


def precision_recall_at_k(recommendations, test_ratings, k=5):
    """
    Calculate precision and recall at k for a given set of recommendations and test ratings.

    Parameters:
    recommendations (dict): A dictionary where the keys are user IDs and the values are lists of recommended items.
    test_ratings (DataFrame): A DataFrame containing the test ratings data, with columns 'user_id' and 'movie_id'.
    k (int): The number of recommendations to consider for each user. Default is 5.

    Returns:
    tuple: A tuple containing the average precision and average recall.
    """

    precisions = []
    recalls = []
    
    for user_id, recs in recommendations.items():
        # Get the test set for this user
        relevant_items = test_ratings[test_ratings.user_id == user_id].movie_id.values
        
        # Calculate precision
        top_k_recs = recs[:k]
        num_relevant_items = sum(item in relevant_items for item in top_k_recs)
        
        precision = num_relevant_items / k
        recall = num_relevant_items / len(relevant_items) if len(relevant_items) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    
    return avg_precision, avg_recall

def ndcg_at_k(recommendations, test_ratings, k=5):
    """
    Calculate the Normalized Discounted Cumulative Gain (NDCG) at a given value of k.

    Parameters:
    recommendations (dict): A dictionary where the keys are user IDs and the values are lists of recommended items.
    test_ratings (DataFrame): A DataFrame containing the test ratings data, with columns 'user_id' and 'movie_id'.
    k (int): The number of recommendations to consider for each user. Default is 5.

    Returns:
    float: The average NDCG across all users.

    """
    all_ndcgs = []
    
    for user_id, recs in recommendations.items():
        # Get the test set for this user
        relevant_items = test_ratings[test_ratings.user_id == user_id].movie_id.values
        
        # Create relevance scores
        relevance = np.isin(recs[:k], relevant_items).astype(int)
        
        # Calculate NDCG
        ideal_relevance = np.sort(relevance)[::-1]
        if np.sum(ideal_relevance) > 0:
            ndcg = ndcg_score([ideal_relevance], [relevance], k=k)
            all_ndcgs.append(ndcg)
    
    return np.mean(all_ndcgs)

def mean_average_precision_at_k(recommendations, test_ratings, k=5):
    """
    Calculate the mean average precision at k for a set of recommendations.

    Parameters:
    recommendations (dict): A dictionary where the keys are user IDs and the values are lists of recommended items.
    test_ratings (DataFrame): A DataFrame containing the test ratings data, with columns 'user_id' and 'movie_id'.
    k (int): The number of recommendations to consider for each user. Default is 5.

    Returns:
    float: The mean average precision at k.

    """
    ap_scores = []
    
    for user_id, recs in recommendations.items():
        # Get the test set for this user
        relevant_items = test_ratings[test_ratings.user_id == user_id].movie_id.values
        
        # Calculate precision at each rank position where a relevant item is found
        top_k_recs = recs[:k]
        num_relevant_items = 0
        score = 0.0
        
        for i, item in enumerate(top_k_recs):
            if item in relevant_items:
                num_relevant_items += 1
                score += num_relevant_items / (i + 1)
        
        if num_relevant_items > 0:
            ap_scores.append(score / num_relevant_items)
    
    return np.mean(ap_scores)


def mean_reciprocal_rank(recommendations, test_ratings):
    """
    Calculate the mean reciprocal rank (MRR) for a set of recommendations.

    Parameters:
    - recommendations (dict): A dictionary where the keys are user IDs and the values are lists of recommended items.
    - test_ratings (DataFrame): A DataFrame containing the test ratings data, with columns 'user_id' and 'movie_id'.

    Returns:
    - mean_rr (float): The mean reciprocal rank score.

    """
    rr_scores = []
    
    for user_id, recs in recommendations.items():
        # Get the test set for this user
        relevant_items = test_ratings[test_ratings.user_id == user_id].movie_id.values
        
        # Calculate reciprocal rank
        for rank, item in enumerate(recs):
            if item in relevant_items:
                rr_scores.append(1 / (rank + 1))
                break
        else:
            rr_scores.append(0.0)
    
    return np.mean(rr_scores)

def hit_rate_at_k(recommendations, test_ratings, k=5):
    """
    Calculate the hit rate at k for a recommendation system.

    Parameters:
    recommendations (dict): A dictionary where the keys are user IDs and the values are lists of recommended items.
    test_ratings (DataFrame): A DataFrame containing the test ratings data, with columns 'user_id' and 'movie_id'.
    k (int): The number of recommendations to consider for each user. Default is 5.

    Returns:
    float: The hit rate at k, which is the proportion of users for whom at least one of the top-k recommendations is relevant.
    """
    hits = []
    
    for user_id, recs in recommendations.items():
        # Get the test set for this user
        relevant_items = test_ratings[test_ratings.user_id == user_id].movie_id.values
        
        # Check if there is a hit in the top-k recommendations
        if any(item in relevant_items for item in recs[:k]):
            hits.append(1)
        else:
            hits.append(0)
    
    return np.mean(hits)


def coverage_at_k(recommendations, k=5, total_items=None):
    """
    Calculate the coverage at k for a set of recommendations.

    Args:
        recommendations (dict): A dictionary where the keys are user IDs and the values are lists of recommended items.
        k (int, optional): The number of recommendations to consider for each user. Default is 5.
        total_items (int, optional): The total number of unique items in the dataset. If not provided, it will be calculated based on the recommendations.

    Returns:
        float: The coverage at k, which is the proportion of unique items recommended across all users.

    """
    recommended_items = set()
    
    for recs in recommendations.values():
        recommended_items.update(recs[:k])
    
    if total_items is None:
        total_items = len(set(item for recs in recommendations.values() for item in recs))
    return len(recommended_items) / total_items


def evaluate_recommender_system(recommendations, test_ratings,total_amount_of_movies, k=5):
    """
    Evaluates a recommender system by calculating various metrics.

    Args:
        recommendations (dict): A dictionary where the keys are user IDs and the values are lists of recommended items.
        test_ratings (DataFrame): A DataFrame containing the test ratings data, with columns 'user_id' and 'movie_id'.
        k (int, optional): The number of recommendations to consider. Defaults to 5.

    Returns:
        dict: A dictionary containing the evaluation metrics.
            - 'Precision@K': Precision at K.
            - 'Recall@K': Recall at K.
            - 'NDCG@K': Normalized Discounted Cumulative Gain at K.
            - 'MAP@K': Mean Average Precision at K.
            - 'MRR': Mean Reciprocal Rank.
            - 'Hit Rate@K': Hit Rate at K.
            - 'Coverage@K': Coverage at K.
    """
    precision, recall = precision_recall_at_k(recommendations, test_ratings, k)
    ndcg = ndcg_at_k(recommendations, test_ratings, k)
    map_k = mean_average_precision_at_k(recommendations, test_ratings, k)
    mrr = mean_reciprocal_rank(recommendations, test_ratings)
    hit_rate = hit_rate_at_k(recommendations, test_ratings, k)
    coverage = coverage_at_k(recommendations, k, total_items=total_amount_of_movies)
    
    return {
        'Precision@K': precision,
        'Recall@K': recall,
        'NDCG@K': ndcg,
        'MAP@K': map_k,
        'MRR': mrr,
        'Hit Rate@K': hit_rate,
        'Coverage@K': coverage
    }
