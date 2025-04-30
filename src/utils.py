import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def plot_grouped_dq(df_all: pd.DataFrame, cols: list[str],
                    reg_col_dfl: str, metric_col_dfl: str,
                    reg_col_lr: str, metric_col_lr: str, metric: str, epsilon: float,
                    title_fontsize: int = 16, label_fontsize: int = 12, tick_fontsize: int = 10):
    """
    Generates a grouped bar chart comparing the mean of two metric
    columns with standard error, across a common regularization parameter,
    with customizable font sizes, handling infinite values.
    """
    df_plot = df_all[cols].copy()
    print(f"Shape of df_plot: {df_plot.shape}")

    df_eps = df_plot[df_plot['epsilon'] == epsilon].copy()

    # Group by reg_col_dfl and get mean and standard error of metric_col_dfl
    grouped_dfl = df_eps.groupby(reg_col_dfl)[metric_col_dfl].agg(['mean', 'sem', 'size']).reset_index()

    # Replace inf and -inf with 0 in the grouped_dfl DataFrame
    grouped_dfl = grouped_dfl.replace([np.inf, -np.inf], 0)

    grouped_dfl = grouped_dfl.rename(columns={reg_col_dfl: 'lambda', 'mean': 'mean_method1', 'sem': 'stderr_method1', 'size': 'size_method1'})

    # Group by reg_col_lr and get mean and standard error of metric_col_lr
    grouped_lr = df_eps.groupby(reg_col_lr)[metric_col_lr].agg(['mean', 'sem', 'size']).reset_index()

    # Replace inf and -inf with 0 in the grouped_lr DataFrame
    grouped_lr = grouped_lr.replace([np.inf, -np.inf], 0)

    grouped_lr = grouped_lr.rename(columns={reg_col_lr: 'lambda', 'mean': 'mean_method2', 'sem': 'stderr_method2', 'size': 'size_method2'})

    # Merge the grouped DataFrames on the 'lambda' column
    merged_grouped = pd.merge(grouped_dfl, grouped_lr, on='lambda', how='outer').sort_values(by='lambda')

    # Prepare data for plotting
    labels = merged_grouped['lambda'].astype(str)
    mean_method1 = merged_grouped['mean_method1'].fillna(0)
    stderr_method1 = merged_grouped['stderr_method1'].fillna(0)
    mean_method2 = merged_grouped['mean_method2'].fillna(0)
    stderr_method2 = merged_grouped['stderr_method2'].fillna(0)

    x = np.arange(len(labels))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(18, 8))

    rects1 = ax.bar(x - bar_width/2, mean_method1, bar_width, yerr=stderr_method1, capsize=5, label=metric_col_dfl)
    rects2 = ax.bar(x + bar_width/2, mean_method2, bar_width, yerr=stderr_method2, capsize=5, label=metric_col_lr, color='orange')

    # Add labels, title, and ticks with specified font sizes
    ax.set_ylabel(f'{metric}', fontsize=label_fontsize)
    ax.set_xlabel('Regularization Parameter (lambda)', fontsize=label_fontsize)
    ax.set_title(f'Comparison of Mean {metric} (epsilon={epsilon})', fontsize=title_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.legend(fontsize=label_fontsize)

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_eps_lambda_csv('eps_finer.csv')