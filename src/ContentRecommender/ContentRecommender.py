import pandas as pd

class ContentRecommender:
    def __init__(self):
        """ Esse classe implementa um recomendador baseado em conteúdo que utiliza os genêros dos filmes
            como features para identificar as similaridades de novos itens com base no que os usuários já
            consumiram.
        """
        pass
    
    def _correctContentData(self):
        """ Essa função faz a correção dos dados de conteúdo dos filmes. É feita correção de tipos, remoção
            de valores nan e N/A. Além disso, ela seleciona apenas as colunas que serão usadas na recomendação
            a fim de facilitar o entendimento dos fatores utilizados.
        """

        self.content_['Metascore'] = self.content_['Metascore'].str.replace('N/A', '1.0', regex=True)
        self.content_['Metascore'] = self.content_['Metascore'].astype(str).astype('float64')

        self.content_['imdbRating'] = self.content_['imdbRating'].str.replace('N/A', '1.0', regex=True)
        self.content_['imdbRating'] = self.content_['imdbRating'].astype(str).astype('float64')

        self.content_['imdbVotes'] = self.content_['imdbVotes'].str.replace('N/A', '1', regex=True)
        self.content_['imdbVotes'] = self.content_['imdbVotes'].str.replace(',', '', regex=True)
        self.content_['imdbVotes'] = self.content_['imdbVotes'].astype(str).astype('float64')

        self.content_['BoxOffice'] = self.content_['BoxOffice'].str.replace('N/A', '1.0', regex=True)
        self.content_['BoxOffice'] = self.content_['BoxOffice'].str.replace(',', '', regex=True)
        self.content_['BoxOffice'] = self.content_['BoxOffice'].str.replace('$', '', regex=True)
        self.content_['BoxOffice'] = self.content_['BoxOffice'].astype(str).astype('float64')

        self.content_['Genre'] = self.content_['Genre'].str.replace(r'[\']', '', regex=True)
        self.content_['Genre'] = self.content_['Genre'].str.replace(' ', '', regex=True)

        self.content_['Metascore'] = self.content_['Metascore'].fillna(0)
        self.content_['imdbRating'] = self.content_['imdbRating'].fillna(0)
        self.content_['imdbVotes'] = self.content_['imdbVotes'].fillna(0)
        self.content_['BoxOffice'] = self.content_['BoxOffice'].fillna(0)

        self.content_ = self.content_.drop(self.content_[(self.content_['Genre'] == 'N/A')].index)

        self.content_ = self.content_[['ItemId','Genre', 'Metascore', 'imdbRating', 'imdbVotes', 'BoxOffice']].copy()

    def _transformGenresIntoColumns(self):
        """ Essa função converte a lista de gêneros que cada item possui em novas colunas. Cada gênero recebe uma
            nova coluna e o item recebe valor 1 se ele possui tal gênero em sua lista de gêneros. Essa conversão é 
            feita para que os gêneros possam ser tratados com features.
        """

        splittedGenres = self.content_['Genre'].str.split(',')
        self.genres_ = list(dict.fromkeys([y for x in splittedGenres for y in x]).keys())

        for genre in self.genres_:
            self.content_[genre] = self.content_['Genre'].apply(lambda x : 1 if genre in x else 0)

        self.content_ = self.content_.drop(['Genre'], axis=1)
    
    def _mergeRawData(self):
        """ Essa função cria novas colunas nas tabelas de itens avaliados e itens para predição relacionadas
            com as features de gênero de cada item da tabela de conteúdo. Os itens sem dados são preenchidos com
            valores nulos, indicando que não se tem conhecimento das características do filme.
        """

        self.ratings_ = self.ratings_.merge(self.content_, on='ItemId')
        self.ratings_ = self.ratings_.drop(['Metascore', 'imdbRating', 'imdbVotes', 'BoxOffice'], axis=1)

        self.targets_ = self.targets_.merge(self.content_, on='ItemId', how='outer')
        self.targets_ = self.targets_.fillna(0)

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
    
    def _calculateGrade(self):
        """ Essa função cria uma nova coluna correspondente a escala de nota que o usuário deu a um filme. Se a nota é 10 a
            escala é 3, indicando uma recomendação perfeita. Se é 9 ou 8 a escala é 2, se é 7 ou 6 a escala é 1 e se for entre
            0 e 5 a escala é zero, representando uma escolha ruim ou não avaliada.
        """
        self.ratings_['Grade'] = self.ratings_.apply(self._replaceRatings, axis=1)
    
    def _generateSimilarities(self):
        """ Essa função uma matriz usuário-item e a similaridade entre eles como a recomendação de Rocchio, ou seja, o cosseno entre
            o vetor de features dos itens e dos usuários. A similaridade representa o quanto um item não consumido é semelhante aos
            outros itens já consumidos pelo usuário.
        """
        self.ratings_['Frequency'] = self.ratings_.groupby('UserId')['UserId'].transform('count')

        for genre in self.genres_:
            self.ratings_[genre] = self.ratings_[genre] * self.ratings_['Grade'] / self.ratings_['Frequency']
        
        userProfile = self.ratings_.copy().drop(['ItemId', 'Timestamp', 'Rating', 'Grade', 'Frequency'], axis=1)
        userProfile = userProfile.groupby('UserId').sum()

        for genre in self.genres_:
            userProfile['User' + genre] = userProfile[genre]
            userProfile = userProfile.drop(genre, axis=1)

        for genre in self.genres_:
            self.targets_['Item' + genre] = self.targets_[genre]
            self.targets_ = self.targets_.drop(genre, axis=1)
        
        self.targets_ = pd.merge(self.targets_, userProfile, on=['UserId'])

        similarities = self.targets_[['UserId', 'ItemId', 'Metascore', 'imdbRating', 'imdbVotes', 'BoxOffice']].copy()

        similarities['UserItemNum'] = 0
        similarities['UserDen'] = 0
        similarities['ItemDen'] = 0

        for genre in self.genres_:
            similarities['UserItemNum'] += self.targets_['User' + genre] * self.targets_['Item' + genre]
            similarities['UserDen'] += self.targets_['User' + genre] * self.targets_['User' + genre]
            similarities['ItemDen'] += self.targets_['Item' + genre] * self.targets_['Item' + genre]

        similarities['UserDen'] = similarities['UserDen'].pow(1./2)
        similarities['ItemDen'] = similarities['ItemDen'].pow(1./2)
        similarities['Similarity'] = similarities['UserItemNum'] / (similarities['UserDen'] * similarities['ItemDen'])

        similarities = similarities.drop(['UserItemNum', 'UserDen', 'ItemDen'], axis=1)

        self.semilarities_ = similarities
    
    def getPredictions(self, ratings, content, targets):
        """ Essa função realiza as chamadas às funções que executam cada passo da recomendação baseada em conteúdo. Ela retorna
            um dataframe que contem os pares usuários-itens e a similaridade entre eles.
        """

        self.content_ = content
        self.ratings_ = ratings
        self.targets_ = targets

        self._correctContentData()
        self._transformGenresIntoColumns()
        self._mergeRawData()
        self._calculateGrade()
        self._generateSimilarities()

        return self.semilarities_


