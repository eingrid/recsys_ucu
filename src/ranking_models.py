import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PairwiseRankingModel(nn.Module):
    def __init__(self):
        super(PairwiseRankingModel, self).__init__()
        # Define the architecture
        self.fc1 = nn.Linear(3 + 19, 64)  # User features (3) + Film1 features (19)
        self.fc2 = nn.Linear(64 + 19, 32)  # Add Film2 features (19)
        self.fc3 = nn.Linear(32, 1)  # Output

    def forward(self, user_features, film1_features, film2_features):
        """
        Predict which movie is better between film1 and film2 for a given user.
        After that we can rank list of movies using majority voting.
        I.E. if we have 5 movies we can compare each movie with the others and count the number of times it was selected as the best movie.
        """
        # Concatenate user features and film1 features
        x = torch.cat([user_features, film1_features], dim=1)
        x = torch.relu(self.fc1(x))
        # Concatenate with film2 features
        x = torch.cat([x, film2_features], dim=1)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ListwiseRankingModel(nn.Module):
    def __init__(self, user_feature_dim=3, movie_feature_dim=19, hidden_dim=64):
        super(ListwiseRankingModel, self).__init__()
        self.user_feature_layer = nn.Linear(user_feature_dim, hidden_dim)
        self.movie_feature_layer = nn.Linear(movie_feature_dim, hidden_dim)
        self.merging_layer = nn.Linear(2 * hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, user_features, movie_features):
        """
        Given user and movie features, compute the relevance score aftter that we can use the scores to rank the movies.
        """
        # Process user features
        user_hidden = F.relu(self.user_feature_layer(user_features))
        
        # Process movie features
        movie_hidden = F.relu(self.movie_feature_layer(movie_features))
        
        # Merge user and movie features
        merged_features = torch.cat((user_hidden.unsqueeze(1).repeat(1, movie_features.size(1), 1), movie_hidden), dim=2)
        merged_hidden = F.relu(self.merging_layer(merged_features))
        
        # Compute scores for each movie
        scores = self.output_layer(merged_hidden)
        return scores.squeeze()

def load_pairwise_model(path="../../artifacts/pairwise_ranking_model.pth", device='cpu'):
    model = PairwiseRankingModel().to(device)  # Instantiate the model
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {path}")
    return model

def load_listwise_model(path="../../artifacts/listwise_ranking_model.pth", device='cpu'):
    model = ListwiseRankingModel().to(device)  # Instantiate the model
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {path}")
    return model

def convert_data_for_listwise(data, user_id, movie_id1, movie_id2, user_features_names : list[str]  = ['gender', 'age', 'occupation'], movie_features_names :  list[str] = ['year', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']):
    """
    Convert data for the listwise model
    """
    user_feat = torch.tensor(data[user_features_names].loc[user_id].values.astype(np.float32), dtype=torch.float32).unsqueeze(0)
    film1_feat = torch.tensor(data[movie_features_names].loc[movie_id1].values.astype(np.float32), dtype=torch.float32).unsqueeze(0)
    film2_feat = torch.tensor(data[movie_features_names].loc[movie_id2].values.astype(np.float32), dtype=torch.float32).unsqueeze(0)
    
    return user_feat, film1_feat, film2_feat
    
def convert_data_for_pairwise(data ,user_id : int, movie_ids : list[int], user_features_names : list[str]  = ['gender', 'age', 'occupation'], movie_features_names :  list[str] = ['year', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'], padding_size = 25) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert data for the pairwise model
    
    Args:
        data (pandas.DataFrame): The input data containing user and movie features.
        user_id (int): The ID of the user.
        movie_ids (list[int]): The IDs of the movies.
        user_features_names (list[str], optional): The names of the user features. Defaults to ['gender', 'age', 'occupation'].
        movie_features_names (list[str], optional): The names of the movie features. Defaults to ['year', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
            'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'].
        padding_size (int, optional): The size of padding to be added to the movie features. Defaults to 25.
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the user features tensor and the movie features tensor.
    """
    if len(movie_ids) > padding_size or len(movie_ids) == 0:
        raise ValueError(f"Number of movies should be less than {padding_size} and higher than 0")
    
    user_features = data[data['user_id'] == user_id][user_features_names].iloc[0].values
    movie_features = []
    for movie_id in movie_ids:
        movie_features.append(data[data['movie_id'] == movie_id][movie_features_names].iloc[0].values)
        
    # Check the shape of movie_features

    # We need to add padding to the movie features if there are less than 25 movies
    num_movies = len(movie_ids)
    num_features_per_movie = len(movie_features_names)

    if num_movies < 25:
        padding_size = 25 - num_movies
        # Create a padding array of shape (padding_size, num_features_per_movie) filled with -1
        padding_array = -1 * np.ones((padding_size, num_features_per_movie))
        # Append the padding array to movie_features
        movie_features = np.vstack((movie_features, padding_array))
    
    
    # Convert to tensors
    user_features_tensor = torch.tensor(user_features.astype(np.float32), dtype=torch.float32).unsqueeze(0)
    movie_features_tensor = torch.tensor(movie_features.astype(np.float32), dtype=torch.float32).unsqueeze(0)
    
    return user_features_tensor, movie_features_tensor