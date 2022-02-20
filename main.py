import sys
import time

import pandas as pd

from src.CollaborativeRecommender.CollaborativeRecommender import CollaborativeRecommender
from src.ContentRecommender.ContentRecommender import ContentRecommender
from src.HybridRecommender.HybridRecommender import HybridRecommender

def main():
    startTime = time.time()

    with open(sys.argv[1], 'r') as f:
        ratings = pd.read_json(f, lines=True)

    with open(sys.argv[2], 'r') as f:
        content = pd.read_json(f, lines=True)
        
    with open(sys.argv[3], 'r') as f:
        targets = pd.read_csv(f, sep=',', engine='python')


    ## Collaborative Filtering Recommender
    training = ratings[['UserId', 'ItemId', 'Rating']].sample(frac=0.8, random_state=8)
    validation = ratings.drop(training.index.tolist())
    cfReccomendations = CollaborativeRecommender()
    cfReccomendations = cfReccomendations.getPredictions(training, validation, targets, saveToFile=False, printOnConsole=False, getPredictions=True)

    ## Content Based Recommender
    cbReccomendations = ContentRecommender()
    cbReccomendations = cbReccomendations.getPredictions(ratings, content, targets)

    ## Hybrid Recommender
    hybridReccomendations = HybridRecommender()
    hybridReccomendations = hybridReccomendations.getPredictions(cfReccomendations, cbReccomendations, saveToFile=True, printOnConsole=False)
        
    print("Time: %s seconds " % (time.time() - startTime))

main()
