import numpy as np
import pandas as pd

def sample_l2lap(eta:float, d:int) -> np.array:
  R = np.random.gamma(d, scale = 1.0 / eta)
  Z = np.random.normal(0, 1, size = d)
  return R * (Z / np.linalg.norm(Z)) #shape is (d,) one dimensional

def generate_synth_data(B_train, B_test, d):
    """
    Generates synthetic data for training and testing.

    Args:
        B_train (int): Number of training samples.
        B_test (int): Number of testing samples.
        d (int): Dimension of the feature vectors.

    Returns:
        tuple: (X_train, y_train, X_test, y_test, b)
            X_train (numpy.ndarray): Training features, shape (B_train, d).
            y_train (numpy.ndarray): Training labels, shape (B_train,).
            X_test (numpy.ndarray): Testing features, shape (B_test, d).
            y_test (numpy.ndarray): Testing labels, shape (B_test,).
            b (numpy.ndarray) : noise vector of shape (d,)
    """
    # Generate random features with L2 norm <= 1
    X_train = np.random.randn(B_train, d)
    X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)

    X_test = np.random.randn(B_test, d)
    X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)

    # Generate random labels (-1 or 1)
    y_train = np.random.choice([-1, 1], size=(B_train,))
    y_test = np.random.choice([-1, 1], size=(B_test,))

    return X_train, y_train, X_test, y_test

def generate_eps_lambda_csv(output_file="eps_lambda.csv"):
    # epsilons = [0.1, 0.5, 1, 2]
    # lamb_dfls = [0.1, 0.5, 1, 5, 10]
    # lamb_lrs = [0.1, 0.5, 1, 5, 10]

    # c_dfl_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
    epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 2]
    lamb_dfls = [0.1, 0.5, 1, 5, 10] 
    lamb_lrs = [0.1, 0.5, 1, 5, 10]

    # Create a list of all possible combinations of the parameters.
    combinations = [(eps, ldfl, llr)
                    for eps in epsilons
                    for ldfl in lamb_dfls
                    for llr in lamb_lrs]

    # Create a Pandas DataFrame from the combinations list.
    df = pd.DataFrame(combinations, columns=['epsilon', 'lambda_dfl', 'lambda_lr'])

    # Save the DataFrame to a CSV file.
    df.to_csv(output_file, index=False)  # index=False prevents writing the DataFrame index to the CSV.
    print(f"Generated parameter combinations and saved to {output_file}")

if __name__ == "__main__":
    generate_eps_lambda_csv('eps_finer.csv')