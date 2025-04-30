import numpy as np
from src.LogLossAccuracy import LogLossAccuracy

class RandomGuess:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_train = X_train.shape[0]
        self.dim = X_train.shape[1]

    def metrics_train(self, lpopt_train):
        lla = LogLossAccuracy(self.X_train, self.y_train)
        ll_rand = lla.get_lossloss_randomguess()
        acc_rand = lla.get_accuracy_randomguess()
        dq_rand = lpopt_train.get_DQ_randomguess()
        return ll_rand, acc_rand, dq_rand
    
    def metrics_test(self, lpopt_test):
        lla = LogLossAccuracy(self.X_test, self.y_test)
        ll_rand = lla.get_lossloss_randomguess()
        acc_rand = lla.get_accuracy_randomguess()
        dq_rand = lpopt_test.get_DQ_randomguess()
        return ll_rand, acc_rand, dq_rand
    
        