class UnsupervisedPreprocessor:
    def __init__(self):

        # ----Statistics needed in preprocessing functions---- #
        self.mean = None
        self.covariance = None
        self.std = None

    def fit(self, data_matrix):
        pass

    def preprocess(self, data_matrix):
        pass

    def __str__(self):
        pass


class SupervisedPreprocessor:
    def __init__(self):
        self.data_matrix = None
        self.labels = None

        # ----Statistics needed in preprocessing functions---- #
        self.mean = None
        self.covariance = None
        self.std = None

    def preprocess(self, data_matrix, labels):
        self.data_matrix = data_matrix
        self.labels = labels

    def __str__(self):
        pass
