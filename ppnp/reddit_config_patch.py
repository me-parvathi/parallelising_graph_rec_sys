import os

# Function to patch the loaders with environment variables
def get_reddit_loader_config():
    batch_size = int(os.environ.get('REDDIT_BATCH_SIZE', '256'))
    num_neighbors = [int(x) for x in os.environ.get('REDDIT_NUM_NEIGHBORS', '5,5').split(',')]
    return batch_size, num_neighbors

# This function will be imported in data.py
