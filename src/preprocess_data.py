from src.project_dirs import *
from src.utils import *
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def split_data(df, target_col:str, train_size=10000, test_size=10000, random_state=42):
    """
    Splits a DataFrame into training and testing sets by taking random samples.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target column.
        train_size (int): The desired size of the training set.
        test_size (int): The desired size of the testing set.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, y_test (pandas DataFrames/Series).
               Returns None if the DataFrame size is less than the combined train and test size.
    """
    if len(df) < train_size + test_size:
        print(f"Warning: The DataFrame size ({len(df)}) is less than the requested combined train ({train_size}) and test ({test_size}) size.")
        return None

    # Separate features (X) and target (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Create the training set
    X_train = X.sample(n=train_size, random_state=random_state)
    y_train = y.loc[X_train.index]

    # Create the testing set by dropping the training samples and then sampling
    X_remaining = X.drop(X_train.index)
    y_remaining = y.drop(y_train.index)

    X_test = X_remaining.sample(n=test_size, random_state=random_state)
    y_test = y_remaining.loc[X_test.index]

    return X_train, X_test, y_train, y_test

def preprocess_adultreconstructed(income_value:int=50000):
    '''
        income_value is used to binarize the target variable
        Preprocess the adult reconstructed dataset
        https://github.com/socialfoundations/folktables/blob/main/adult_reconstruction.csv
    '''
    df = pd.read_csv(str(raw_data_root() / "adult_reconstruction.csv"))
    print("All columns", df.columns)
    print("Original shape", df.shape)
    print("Missing values in columns", df.isnull().sum())
    df.dropna(inplace=True)
    print("Shape after dropping NA", df.shape)

    target_col = 'income'
    numeric_feat = ['hours-per-week', 'age', 'capital-gain', 'capital-loss', 'education-num']
    categorical_feat = ['occupation', 'workclass', 'education', 'marital-status', 'relationship', \
                        'race', 'gender', 'native-country']
    targets = df[target_col]
    df = df.drop(columns=[target_col])

    scaler = MinMaxScaler()
    df[numeric_feat] = scaler.fit_transform(df[numeric_feat])
    df = pd.get_dummies(df, columns=categorical_feat, drop_first=True) # one-hot encoding dimension is #of classes
    df = 1.0 * df # convert bools to 0-1
    M = max(np.linalg.norm(df.to_numpy(), ord=2, axis=1)) # get max L2 norms of rows, TODO maybe alternatives here?
    print("Max L2 norm of rows", M)
    df = df / M

    targets = targets.apply(lambda x: 1 if x >= income_value else -1)
    df[target_col] = targets # add target column back to df
    save_path = processed_data_root() / "adult_reconstructed.csv"
    df.to_csv(save_path, index=False)

def preprocess_adult_uci(): # there is slight variation in this dataset, some missing vals, use preprocess_adultreconstructed
    pass

if __name__=="__main__":
    preprocess_adultreconstructed()
    df = pd.read_csv(str(processed_data_root() / "adult_reconstructed.csv"))
    print(df.head())
    X_train, X_test, y_train, y_test = split_data(df, target_col='income', train_size=1000, test_size=1000)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    lamb = 0.1
    ts_logreg = TwoStage(X_train, y_train, X_test, y_test)
    w_nopri = ts_logreg.train_noprivacy(lamb=lamb)
    
    


