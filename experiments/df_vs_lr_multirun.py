from src.LogLossAccuracy import LogLossAccuracy
from src.DFclosedform import DFL
from src.TwoStageLogReg import TwoStage
from src.LPopt import LPOpt
from src.project_dirs import processed_data_root, output_dir
from src.preprocess_data import split_data
import numpy as np
import pandas as pd
from tqdm import tqdm


def df_vs_lr(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
             epsilon: float, c_dfl: float, lamb_dfl: float, lamb_lr: float, num_runs: int = 100):
    '''
        epsilon: privacy parameter
        c_dfl: Q = c_dfl * I in the DFL relaxation
        lambda_dfl: is the regularization for the DFL obj function
        lambda_lr: is the regularization for the Logreg obj function
    '''
    
    assert X_train.shape[0] == y_train.shape[0] == X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    lpopt_train = LPOpt(X=X_train, y=y_train) # the optimization proble max y^T z, Gz\leq h, here we take z_i \in [0,1] default cosntraints
    lpopt_test = LPOpt(X=X_test, y=y_test)
    results = []
    for i in range(num_runs): # noise for privacy is random so we measure across multiple runs
        # Decision focused learning
        dfl = DFL(X_train, y_train, X_test, y_test)
        w_dfl = dfl.train_privately(epsilon=epsilon, c=c_dfl, lamb=lamb_dfl)
        # loss, acc, decision quality
        trainloss_dfl, trainacc_dfl, traindq_dfl = dfl.metrics_train(w_dfl, lpopt_train)
        testloss_dfl, testacc_dfl, testdq_dfl = dfl.metrics_test(w_dfl, lpopt_test)

        # Logistic regression
        logreg = TwoStage(X_train, y_train, X_test, y_test)
        w_lr = logreg.train_privately(epsilon=epsilon, lamb=lamb_lr)
        trainloss_lr, trainacc_lr, traindq_lr = logreg.metrics_train(w_lr, lpopt_train)
        testloss_lr, testacc_lr, testdq_lr = logreg.metrics_test(w_lr, lpopt_test)

        di = {'run': i,
            'key': f"eps{epsilon}_c{c_dfl}_lambdfl{lamb_dfl}_lamblr{lamb_lr}",
            'epsilon': epsilon,
            'lamb_dfl': lamb_dfl,
            'c_dfl': c_dfl,
            'lamb_lr': lamb_lr,
            'w_dfl': np.linalg.norm(w_dfl, ord=2),
            'w_lr': np.linalg.norm(w_lr, ord=2),
            'trainloss_dfl': trainloss_dfl,
            'trainacc_dfl': trainacc_dfl,
            'traindq_dfl': traindq_dfl,
            'testloss_dfl': testloss_dfl,
            'testacc_dfl': testacc_dfl,
            'testdq_dfl': testdq_dfl,
            'trainloss_lr': trainloss_lr,
            'trainacc_lr': trainacc_lr,
            'traindq_lr': traindq_lr,
            'testloss_lr': testloss_lr,
            'testacc_lr': testacc_lr,
            'testdq_lr': testdq_lr
        }
        results.append(di)
    return pd.DataFrame(results)

if __name__ == "__main__":
    filepath = str(processed_data_root() / "adult_recon26000.csv")
    df = pd.read_csv(filepath)
    B = 1000 # train and test sizes, same
    num_runs = 100
    target_col = 'income'
    X_train, X_test, y_train, y_test = split_data(df=df, target_col=target_col, train_size=B, test_size=B, random_state=42)
    X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values

    # this below can be parallelized
    epsilons = [0.1, 0.5, 1, 2]
    c_dfls = [0.01, 0.1, 1]
    lamb_dfls = [0.1, 0.5, 1, 5, 10]
    lamb_lrs = [0.1, 0.5, 1, 5, 10]

    # Create a list of all the innermost loop parameters to iterate over with tqdm
    all_combinations = [(epsilon, c_dfl, lamb_dfl, lamb_lr)
                        for epsilon in epsilons
                        for c_dfl in c_dfls
                        for lamb_dfl in lamb_dfls
                        for lamb_lr in lamb_lrs]

    for epsilon, c_dfl, lamb_dfl, lamb_lr in tqdm(all_combinations, desc="Parameter Sweep"):
        resdf = df_vs_lr(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                        epsilon=epsilon, c_dfl=c_dfl, lamb_dfl=lamb_dfl, lamb_lr=lamb_lr, num_runs=num_runs)
        save_filename = f"df_vs_lr_multisamples_B{B}_eps{epsilon}_c{c_dfl}_lambdfl{lamb_dfl}_lamblr{lamb_lr}.csv"
        resdf.to_csv(str(output_dir() / save_filename), index=False)
        