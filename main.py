import sys
import time

import pandas as pd

from src.CollaborativeRecommender.CollaborativeRecommender import CollaborativeRecommender

def main():
    startTime = time.time()

    with open(sys.argv[1], 'r') as f:
        ratings1 = pd.read_json(f, lines=True)
        ratings = ratings1[['UserId', 'ItemId', 'Rating']]

    with open(sys.argv[2], 'r') as f:
        content = pd.read_json(f, lines=True)
        
    with open(sys.argv[3], 'r') as f:
        targets = pd.read_csv(f, sep=',', engine='python')


    ## Collaborative Recommender
    training = ratings.sample(frac=0.8, random_state=8)
    validation = ratings.drop(training.index.tolist())

    recommender = CollaborativeRecommender()

    # Caso seja necessário apenas printar na tela os valores ou gerar um arquivo, modifique os parametros saveToFile e printOnConsole
    recommender.runRecommender(training, validation, targets, startTime, saveToFile=True, printOnConsole=False)
        
    print("Time: %s seconds " % (time.time() - startTime))

main()
