import gurobipy as gp
from gurobipy import GRB
import numpy as np

class LPOpt:
    def __init__(self, X_test, y_test, G, h):
        self.X_test = X_test
        self.y_test = y_test
        self.n_test, self.dim = X_test.shape
        self.G = G
        self.h = h

    def get_z(self, y_hat):
        '''
            max_z y_hat^T z
            s.t. G z <= h

            y_hat is the predicted label, shape (n_test,)
            G is the constraint matrix, shape (m, n_test)
            h is the constraint vector, shape (m,)
        '''
        try:
            model = gp.Model("LP_z")
            n_test = y_hat.shape[0]
            z = model.addMVar(shape=(n_test,), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="z")
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
            raise ValueError("Weight vector w must have the same dimension as the feature matrix X_test.")
        y_hat = self.X_test @ w
        z_hat, _ = self.get_z(y_hat)
        if z_hat is not None:
            self.DQ = self.y_test @ z_hat
            return self.DQ
        else:
            return None

if __name__ == '__main__':
    n_test = 5
    dim = 3
    m = 4

    X_test = np.random.rand(n_test, dim)
    y_test = np.random.rand(n_test)


    G = np.vstack((np.identity(n_test), -np.identity(n_test)))
    h = np.concatenate((np.ones(n_test), np.zeros(n_test)))
    lp_opt = LPOpt(X_test, y_test, G, h)

    zhat, obj = lp_opt.get_z(y_hat=np.array([-1,1,2,-1,-2]))
    print(zhat, obj)

    # Example weight vector
    w = np.random.normal(scale=10,size=(dim,)) # dummy

    # Get the optimal z for a given y_hat
    # y_hat_example = X_test @ w
    # z_optimal = lp_opt.get_z(y_hat_example)
    # print("Optimal z:\n", z_optimal)

    # Compute the decision quality
    dq = lp_opt.get_DQ(w)
    print("Decision Quality (DQ):", dq)