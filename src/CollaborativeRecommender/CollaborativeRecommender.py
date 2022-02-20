import numpy as np
import pandas as pd

class CollaborativeRecommender:
    def __init__(self, learningRate=0.01, regularizationFactor=0.05, nEpochs=50, nFactors=25, stopThreshold=0.000001):
        """ Esse classe implementa um recomendador colaborativo baseado no modelo de fator latente utilizando da ideia
            da decomposição em valores singulares (SVD). Os valores pré-definidos na chamada da função representam
            a melhor configuração encontrada após testes.

        Attributes:
        -----------
            dataFrame (pandas dataframe): Dados de treino com as notas de usuário para itens.

            learningRate (float): Taxa de aprendizado do sistema.

            regularizationFactor (float): 

            nEpochs (int): Número de passos para o SGD (Stochastic Gradient Descent)

            nFactors (int): Número de fatores latentes do sistema

            stopThreshold (float): Caso a diferença do RMSE de um novo passo em relação ao anterior seja menor que
            esse limiar, o recomendador para de realizar o treino visto que está saturado

            minRating (int): Maior nota permitida a um item

            maxRating (int): Menor nota permitida a um item

        """
        
        self.learningRate = learningRate
        self.regularizationFactor = regularizationFactor
        self.nEpochs = nEpochs
        self.nFactors = nFactors
        self.stopThreshold = stopThreshold
        self.minRating = 0
        self.maxRating = 10

    def _initBias(self, dataFrame):
        """ Esse método inicializa o vetor de bias de itens e de usuários. 

        Parameters:
        -----------
            dataFrame (pandas dataframe): Dados de treino com as notas de usuário e item.

        Returns:
        -----------
            bu (array): bias dos usuários, inicialmente zerados

            bi (array): bias dos itens, inicialmente zerados

        """
        nUsers = len(np.unique(dataFrame[:, 0]))
        nItems = len(np.unique(dataFrame[:, 1]))

        bu = np.zeros(nUsers)
        bi = np.zeros(nItems)

        return bu, bi

    def _initLatentFactors(self, dataFrame):
        """ Esse método inicializa as matrizes de fator latente de itens e de usuários. 

        Parameters:
        -----------
            dataFrame (pandas dataframe): Dados de treino com as notas de usuário para itens.

        Returns:
        -----------
            pu (numpy array): matriz de fator latente dos usuários, inicialmente preenchidos com 
            números aleatórios gerados a partir de uma distribuição normal

            qi (numpy array): matriz de fator latente dos itens, inicialmente preenchidos com 
            números aleatórios gerados a partir de uma distribuição normal

        """
        nUsers = len(np.unique(dataFrame[:, 0]))
        nItems = len(np.unique(dataFrame[:, 1]))

        np.random.seed(seed=13)
        pu = np.random.normal(0, .01, (nUsers, self.nFactors))
        qi = np.random.normal(0, .01, (nItems, self.nFactors))

        return pu, qi

    def _initDataSetMapping(self, dataFrame, columnName):
        """ Esse método realiza o mapeamento de uma coluna do dataset para seu relativo indice inteiro

        Parameters:
        -----------
            dataFrame (pandas dataframe): Dataset que contem os dados a serem mapeados
            
            columnName (pandas dataframe): Coluna do dataset que contem os dados a serem mapeados
            
        Returns:
        -----------
            mapping (dict): Dicionário para o mapeamento feito entre os dados e os seus indices inteiros

        """
        userIds = dataFrame[columnName].unique().tolist()
        nUsers = len(userIds)
        userIndexes = range(nUsers)
        
        mapping = dict(zip(userIds, userIndexes))
        return mapping

    def _generateMappedDataset(self, dataSet):
        """ Esse método gera um novo data set com as tuplas (usuário, item, nota) com os ids dos usuários
            e dos itens convertidos para seus respectivos índices em número inteiro

        Parameters:
        -----------
            dataSet (pandas dataframe): Dataset a ser convertido

        Returns:
        -----------
            mappedDataSet(numpy array): Dataset convertido para seu equivalente com ids inteiros

        """
        mappedDataSet = dataSet.copy()            

        mappedDataSet['UserId'] = mappedDataSet['UserId'].map(self.userMapping_)
        mappedDataSet['ItemId'] = mappedDataSet['ItemId'].map(self.itemMapping_)

        mappedDataSet.fillna(-1, inplace=True)

        mappedDataSet['UserId'] = mappedDataSet['UserId'].astype(np.int32)
        mappedDataSet['ItemId'] = mappedDataSet['ItemId'].astype(np.int32)

        return mappedDataSet[['UserId', 'ItemId', 'Rating']].to_numpy()
    
    def _learnFactors(self):
        """ Esse método executa uma versão da ideia do SGD (Stochastic Gradient Descent) para treino dos 
            fatores latentes de usuários e itens. A ideia é em cada passada os valores de predição de
            testing se aproximem da nota real em validations.  A convergência é detectada pelo RMSE, 
            caso ele não esteja melhorando mais que o threshhold definido, o algoritmo para, mas caso 
            a execução ultrapasse de 4 min o aprendizado termina

        Attributes:
        -----------
            bu_ (array): Bias de usuários que representa a sua tendência de atribuir notas

            bi_ (array): Bias de itens que representa a sua tendência de receber uma nota

            pu_ (array): Matriz de fatores latentes para os usuários

            qi_ (array): Matriz de fatores latentes para os itens

        """
        bu, bi = self._initBias(self.training_)
        pu, qi = self._initLatentFactors(self.training_)

        for epoch in range(self.nEpochs):
            for i in range(self.training_.shape[0]):
                user = int(self.training_[i, 0])
                item = int(self.training_[i, 1])
                rating = self.training_[i, 2]

                prediction = self.globalMean_ + bu[user] + bi[item]

                for factor in range(self.nFactors):
                    prediction += pu[user, factor] * qi[item, factor]

                error = rating - prediction
                bu[user] += self.learningRate * (error - self.regularizationFactor * bu[user])
                bi[item] += self.learningRate * (error - self.regularizationFactor * bi[item])

                for factor in range(self.nFactors):
                    userFactor = pu[user, factor]
                    itemFactor = qi[item, factor]

                    pu[user, factor] += self.learningRate * (error * itemFactor - self.regularizationFactor * userFactor)
                    qi[item, factor] += self.learningRate * (error * userFactor - self.regularizationFactor * itemFactor)
                
            if self._checkIfPredictionsAreImproving(bu, bi, pu, qi, epoch) == False:
                break

        self.bu_ = bu
        self.bi_ = bi
        self.pu_ = pu
        self.qi_ = qi

    def _computeValidationRMSE(self, bu, bi, pu, qi):
        """ Esse método calcula do RMSE de uma instância do recomendador para um dado conjunto de fatores latentes,
            tendo em vista as notas conhecidas no conjunto de validação.

        Returns:
        -----------
            rmse (float): O RMSE (Root-Mean-Square Error) para um conjunto de fatores latentes 

        """
        
        error = []

        for i in range(self.validation_.shape[0]):
            user = int(self.validation_[i, 0])
            item = int(self.validation_[i, 1])
            rating = self.validation_[i, 2]
            prediction = self.globalMean_

            if user > -1:
                prediction += bu[user]

            if item > -1:
                prediction += bi[item]

            if user > -1 and item > -1:
                for factor in range(self.nFactors):
                    prediction += pu[user, factor] * qi[item, factor]

            error.append(rating - prediction)
            
        error = np.array(error)
        rmse = np.sqrt((np.power(error,2)).mean())

        return rmse

    def _checkIfPredictionsAreImproving(self, bu, bi, pu, qi, epoch):
        """ Esse método verifica se o RMSE está melhorando a cada passada do SGD

        Attributes:
        -----------
            bu (array): Bias de usuários que representa a sua tendência de atribuir notas

            bi (array): Bias de itens que representa a sua tendência de receber uma nota

            pu (array): Matriz de fatores latentes para os usuários

            qi (array): Matriz de fatores latentes para os itens

            epoch (int): Atual passo do SGD (Stochastic Gradient Descent)

        
        Returns:
        -----------
            True (bool): se o RMSE melhorou
            False (bool): se o RMSE não melhorou

        """
        
        self.validationRMSE_[epoch] = self._computeValidationRMSE(bu, bi, pu, qi)

        if epoch > 0 and self.validationRMSE_[epoch] + self.stopThreshold > self.validationRMSE_[epoch-1]:
            return False

        return True

    def _makePredictions(self):
        """ Após o aprendizado dos fatores latentes para as matrizes de usuários e itens, esse método realiza a
            predição de notas para o dado conjunto de targets. Caso a nota predita ultrapasse o valor maximo ou
            fique menor que o valor mínimo, é considerado os valores máximo ou mínimo como a nota predita, ao
            invés da nota 

        Attributes:
        -----------
            prediction (float): Nota predita a partir do produto entre as matrizes de fatores latentes para
            usuários e itens

        """
        for user, item in self.targets_:
            knownUser, knownItem = False, False
            prediction = self.globalMean_

            if user in self.userMapping_:
                knownUser = True
                userIndex = self.userMapping_[user]
                prediction += self.bu_[userIndex]

            if item in self.itemMapping_:
                knownItem = True
                itemIndex = self.itemMapping_[item]
                prediction += self.bi_[itemIndex]

            if knownUser and knownItem:
                prediction += np.dot(self.pu_[userIndex], self.qi_[itemIndex])

            if prediction > self.maxRating:
                prediction = self.maxRating
            
            if prediction < self.minRating:
                prediction = self.minRating

            self.predictions_.append([user, item, prediction])

    def getPredictions(self, training, validation, targets, saveToFile=True, printOnConsole=True, getPredictions=True):
        """ Esse método propriamente invoca os outros métodos da classe para processar os dados de
            entrada e devidamente gerar as recomendações de itens em forma de arquivo ou na saída
            padrão.

        Parameters:
        -----------
            training (pandas dataframe): Dados de treino com as notas de usuário para itens.
            
            validation (pandas dataframe): Dados de validação com as notas de usuário para itens.
            
            targets (pandas dataframe): Dados com os usuários e os itens que desejamos realizar as previsões 
            de notas
            
            saveToFile (bool): Se verdadeiro indica que queremos gerar um arquivo .csv de saída com as 
            predições de notas
            
            printOnConsole (bool): Se verdadeitom indica que queremos que a as predições de notas sejam 
            impressas na saída padrão, ou seja, no terminal de execução

        Attributes:
        -----------
            userMapping_ (dict): Dicionário para os ids e o index de int para os usuários

            itemMapping_ (dict): Dicionário para os ids e o index de int para os itens

            training_ (numpy array): Dados de treino do algoritmo com as ids de usuários e itens mapeados para 
            números inteiros
            
            validation_ (numpy array): Dados de validação do algoritmo com as ids de usuários e itens mapeados 
            para números inteiros
            
            targets_ (zip): Iterador para os dados de teste do algoritmo

            predictions_ (array): Notas preditas para cada (usuário, item) em targets_

            globalMean_ (float): Média global das notas do conjunto de treino

            validationRMSE_ (array): RMSE das presdições feitas para o conjunto de validação, usado para
            identificar se as predições não estão melhorando significamente mais

        """

        self.userMapping_ = self._initDataSetMapping(training, columnName='UserId')
        self.itemMapping_ = self._initDataSetMapping(training, columnName='ItemId')

        self.training_ = self._generateMappedDataset(training)
        self.validation_ = self._generateMappedDataset(validation)

        self.targets_ = zip(targets['UserId'], targets['ItemId'])

        self.predictions_ = []
        self.globalMean_ = np.mean(self.training_[:, 2])
        self.validationRMSE_ = np.zeros((self.nEpochs, 1), dtype=float)

        self._learnFactors()
        self._makePredictions()

        if saveToFile:
            self.predictions_ = pd.DataFrame(self.predictions_, columns=['UserId', 'ItemId', 'Predictions'])
            self.predictions_ = self.predictions_.sort_values(['UserId','Predictions'], ascending=[True, False])

            submissionFile = self.predictions_[['UserId','ItemId', 'Predictions']]
            submissionFile.to_csv('submission.csv', index=False, sep=',')
        
        if printOnConsole:
            print('UserId','ItemId')
            for user, item in self.predictions_:
                print(user, item, sep=',')

        if getPredictions:
            self.predictions_ = pd.DataFrame(self.predictions_, columns=['UserId', 'ItemId', 'Predictions'])
            self.predictions_ = self.predictions_.sort_values(['UserId','Predictions'], ascending=[True, False])
            return self.predictions_