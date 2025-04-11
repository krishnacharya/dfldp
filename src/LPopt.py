import gurobipy as gp
from gurobipy import GRB
import numpy as np

class LPOpt:
    def __init__(self, X, y, G=None, h=None):
        self.X, self.y, self.n, self.dim = X, y, X.shape[0], X.shape[1]
        self.G = np.vstack((np.identity(self.n), -np.identity(self.n))) if G is None else G
        self.h = np.concatenate((np.ones(self.n), np.zeros(self.n))) if h is None else h

    def get_z(self, y_hat):
        '''
            max_z y_hat^T z
            s.t. G z <= h

            y_hat is the predicted label, shape (n,)
            G is the constraint matrix, shape (m, n)
            h is the constraint vector, shape (m,)
        '''
        try:
            model = gp.Model("LP_z")
            model.Params.OutputFlag = 0 # suppress Gurobi output
            n = y_hat.shape[0]
            z = model.addMVar(shape=(n,), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="z")
            obj = y_hat @ z
            model.setObjective(obj, GRB.MAXIMIZE)
            constraints = self.G @ z <= self.h
            model.addConstr(constraints, name="constraints")
            model.optimize()

            if model.status == GRB.OPTIMAL:
                z_optimal = z.x
                objective_value = model.objVal
                return z_optimal, objective_value
            else:
                print(f"Optimization problem did not find an optimal solution. Status: {model.status}")
                return None, None

        except gp.GurobiError as e:
            print(f"Error during Gurobi optimization: {e}")
            return None, None
        
    def get_DQ(self, w):
        '''
            Compute the decision quality given the weight vector w
            w is the weight vector, shape (dim,)
            Decision quality given this weight vector w
        '''
        if w.shape[0] != self.dim:
            raise ValueError("Weight vector w must have the same dimension as the feature matrix X.")
        y_hat = self.X @ w
        z_hat, _ = self.get_z(y_hat)
        if z_hat is not None:
            self.DQ = self.y @ z_hat
            return self.DQ
        else:
            return None

if __name__ == '__main__':
    n = 5
    dim = 3

    X = np.random.rand(n, dim)
    y = np.random.rand(n)


    # G = np.vstack((np.identity(n), -np.identity(n)))
    # h = np.concatenate((np.ones(n), np.zeros(n)))
    lp_opt = LPOpt(X, y)

    zhat, obj = lp_opt.get_z(y_hat=np.array([-1,1,2,-1,-2]))
    print(zhat, obj)

    # Example weight vector
    w = np.random.normal(scale=10,size=(dim,)) # dummy

    # Get the optimal z for a given y_hat
    # y_hat_example = X @ w
    # z_optimal = lp_opt.get_z(y_hat_example)
    # print("Optimal z:\n", z_optimal)

    # Compute the decision quality
    dq = lp_opt.get_DQ(w)
    print("Decision Quality (DQ):", dq)