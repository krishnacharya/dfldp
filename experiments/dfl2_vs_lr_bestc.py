import numpy as np
import pandas as pd
from typing import List
from src.LogLossAccuracy import LogLossAccuracy
from src.DFclosedform import DFLv2
from src.TwoStageLogReg import TwoStage
from src.LPopt import LPOptv2
from src.project_dirs import processed_data_root, output_dir_name
from src.preprocess_data import split_data
import numpy as np
import pandas as pd
from tqdm import tqdm

def df_vs_lr(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, \
             epsilon: float, c_values: List[float], lamb_dfl: float, lamb_lr: float, num_runs: int = 100) -> pd.DataFrame:
    '''
        epsilon: privacy parameter
        c_values: A list of c values to search over for DFL
        lambda_dfl: is the regularization for the DFL obj function
        lambda_lr: is the regularization for the Logreg obj function
        num_runs: number of times to run the experiment
    '''
    assert X_train.shape[0] == y_train.shape[0] == X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    results = []
    lpopt_train = LPOptv2(X=X_train, y=y_train)
    lpopt_test = LPOptv2(X=X_test, y=y_test)
    for i in range(num_runs):
        best_w_dfl = None
        best_c_dfl = None
        best_dq_dfl = -np.inf
        save_trainloss_dfl = None # save this for the best c on test_dq_dfl
        save_trainacc_dfl = None
        save_testloss_dfl = None
        save_testacc_dfl = None
        # Decision focused learning
        for c_dfl in c_values:
            dfl = DFLv2(X_train, y_train, X_test, y_test)
            w_dfl = dfl.train_privately(epsilon=epsilon, c=c_dfl, lamb=lamb_dfl)
            trainloss_dfl, trainacc_dfl, traindq_dfl = dfl.metrics_train(w_dfl, lpopt_train)
            testloss_dfl, testacc_dfl, testdq_dfl = dfl.metrics_test(w_dfl, lpopt_test)
            if testdq_dfl > best_dq_dfl:
                save_trainloss_dfl = trainloss_dfl
                save_trainacc_dfl = trainacc_dfl
                save_traindq_dfl = traindq_dfl
                save_testloss_dfl = testloss_dfl
                save_testacc_dfl = testacc_dfl
                best_dq_dfl = testdq_dfl
                best_w_dfl = w_dfl
                best_c_dfl = c_dfl
        # Logistic regression
        logreg = TwoStage(X_train, y_train, X_test, y_test)
        w_lr = logreg.train_privately(epsilon=epsilon, lamb=lamb_lr)
        trainloss_lr, trainacc_lr, traindq_lr = logreg.metrics_train(w_lr, lpopt_train)
        testloss_lr, testacc_lr, testdq_lr = logreg.metrics_test(w_lr, lpopt_test)
        di = {
            'run': i,
            'key': f"eps{epsilon}_lambdfl{lamb_dfl}_lamblr{lamb_lr}",
            'epsilon': epsilon,
            'lamb_dfl': lamb_dfl,
            'c_dfl': best_c_dfl,
            'lamb_lr': lamb_lr,
            'w_dfl': np.linalg.norm(best_w_dfl, ord=2),
            'w_lr': np.linalg.norm(w_lr, ord=2),
            
            'trainloss_dfl': save_trainloss_dfl,
            'trainacc_dfl': save_trainacc_dfl,
            'traindq_dfl': save_traindq_dfl,

            'testloss_dfl': save_testloss_dfl,
            'testacc_dfl': save_testacc_dfl,
            'testdq_dfl': best_dq_dfl,
            
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

    c_dfl_values = [0.01, 0.1, 0.5, 1] # finer grid here [0.01, 0.02 ...5]
    epsilons = [0.1, 0.5, 1, 2] # check more values < 1
    lamb_dfls = [0.1, 0.5, 1, 5, 10]
    lamb_lrs = [0.1, 0.5, 1, 5, 10]
    # diagonose w norm for DFL, epsilon large for term1, c=1
    save_filename_prefix = 'dfl_best_c'
    all_combinations = [(epsilon, lamb_dfl, lamb_lr)
                        for epsilon in epsilons
                        for lamb_dfl in lamb_dfls
                        for lamb_lr in lamb_lrs]

    for epsilon,  lamb_dfl, lamb_lr in tqdm(all_combinations, desc="Parameter Sweep"):
        resdf = df_vs_lr(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                        epsilon=epsilon, c_values = c_dfl_values, lamb_dfl=lamb_dfl, lamb_lr=lamb_lr, num_runs=num_runs)
        save_filename = f"{save_filename_prefix}_{B}_eps{epsilon}_lambdfl{lamb_dfl}_lamblr{lamb_lr}.csv"
        resdf.to_csv(str(output_dir_name('vdebug') / save_filename), index=False)
