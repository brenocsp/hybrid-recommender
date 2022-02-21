import pandas as pd

class HybridRecommender:
    def __init__(self):
        """ Esse classe implementa um recomendador híbrido que combina prediçãos baseada em conteúdo, predições colaborativas e 
            features não personalizadas.
        """
        pass

    def _replaceRatings(self, row):
        """ Essa função substitui uma nota pela escala correspondente a essa nota conforme a métrica estabelecida.
        """
        if row['Rating'] == 10: 
            grade = 3
        elif row['Rating'] > 7: 
            grade = 2
        elif row['Rating'] > 5: 
            grade = 1
        else:
            grade = 0
        return grade

    def _ajustCollaborativeData(self):
        """ Essa função ajusta as notas preditas para a escala de notas positivas e negativas (entre 0 e 4)
        """
        self.cfPredictions_['Rating'] = self.cfPredictions_['Predictions']
        self.cfPredictions_['PredictionGrade'] = self.cfPredictions_.apply(self._replaceRatings, axis=1)
        self.cfPredictions_ = self.cfPredictions_.drop(['Predictions', 'Rating'], axis=1)
    
    def getPredictions(self, cfPredictions, cbPredictions, saveToFile=False, printOnConsole=True):
        """ Essa função realiza a união das recomendações colaborativas, de conteúdo e não personalizadas para gerar
            recomendações para o usuário.
        """

        self.cfPredictions_ = cfPredictions
        self.cbPredictions_ = cbPredictions

        self._ajustCollaborativeData()

        self.reccomendations_ = pd.merge(self.cfPredictions_ , self.cbPredictions_, on=['UserId', 'ItemId'])
        self.reccomendations_['Similarity'] = self.reccomendations_['Similarity']*self.reccomendations_['imdbVotes']
        self.reccomendations_ = self.reccomendations_.sort_values(['UserId','Similarity','imdbVotes','BoxOffice','Metascore','imdbRating','PredictionGrade'], ascending=[True, False, False, False, False, False, False])
        output = self.reccomendations_[['UserId', 'ItemId']].copy()

        if saveToFile:
            output.to_csv('submission.csv', index=False, sep=',')
        
        if printOnConsole:
            pd.set_option("display.max_rows", None, "display.max_columns", None)
            output = output.to_numpy()
            print('UserId,','ItemId')
            for user, item in output:
                print(user, item, sep=',')

