
from utils import sample_l2lap

class DFL:
    def __init__(self, X_train, y_train, X_test, y_test, lamb):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.lamb = lamb
        self.n_train = X_train.shape[0]
        self.dim = X_train.shape[1]
    
    def train_privately(self, epsilon:float, c:float):
        '''
            Pertubed Obj min_w \left[ -y_train^T (Q_inv (X_train w - G^T \gamma_dual)) \right] + 0.5 * lamb * ||w||^2 + 1/n * (b^T w)
            
            We assume Q = c I_n, so Q_inv = 1/c I_n

            Pertubed Obj has a closed form solution:
            theta_dpfl = 1/lamb (1/c * X_train^T y_train - 1/n * b)
        '''
        eta = c * epsilon / (4 * self.n_train**1.5)
        b = sample_l2lap(eta = eta, d = self.dim)
        self.w_pri = 
        return self.w_pri

        