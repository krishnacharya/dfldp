import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class LogLossAccuracy:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def get_lossloss_randomguess(self):
        '''
            Guess label -1, +1 randomly for each sample in X and computes logistic loss
        '''
        y_rand = np.random.choice([-1, 1], size=self.X.shape[0])
        z = y_rand * self.y
        loss = np.mean(np.log(1 + np.exp(-z)))
        return loss

    def get_accuracy_randomguess(self):
        '''
            Guess label -1, +1 randomly for each sample in X and computes accuracy
        '''
        y_rand = np.random.choice([-1, 1], size=self.X.shape[0])
        acc = np.mean(y_rand == self.y)
        return acc

    def get_logloss(self, w):
        '''
            Computes
            1/n * \sum_{i=1}^n_test log(1+exp(-y_i * w^T * x_i))
        '''
        z = self.y * np.dot(self.X, w)
        loss = np.mean(np.log(1 + np.exp(-z)))
        return loss

    def classify(self, w, threshold=0.5):
        '''
            Computes
            predicted label for each sample in X_test

            P(Y=1 | x_i) = 1 / (1 + exp(-w^T * x_i))
            P(Y=-1 | x_i) = 1 - P(Y=1 | x_i)
            predicted label = 1 if P(Y=1 | x_i) > threshold else -1
        '''
        probabilities = 1 / (1 + np.exp(-np.dot(self.X, w)))
        predictions = np.where(probabilities > threshold, 1, -1)
        return predictions

    def get_accuracy(self, w):
        '''
            Computes
            1/n * \sum_{i=1}^n I(y_i == w^T * x_i)
        '''
        preds = self.classify(w)
        return (preds == self.y).mean()

if __name__ == "__main__":
    # Generate some synthetic binary classification data
    X, y = make_classification(n_samples=200, n_features=5, n_informative=3, n_redundant=0,
                               random_state=42)
    y = np.where(y == 0, -1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    log_loss_accuracy_obj = LogLossAccuracy(X_test, y_test)
    # Initialize a random weight vector (in a real scenario, this would be learned)
    num_features = X_train.shape[1]
    initial_weights = np.random.rand(num_features)
    # Calculate the log loss with the initial weights
    logloss = log_loss_accuracy_obj.get_logloss(initial_weights)
    print(f"Log Loss on the test set with random weights: {logloss:.4f}")
    accuracy = log_loss_accuracy_obj.get_accuracy(initial_weights)
    print(f"Accuracy on the test set with random weights: {accuracy:.4f}")

    # You would typically train a model (e.g., using gradient descent) to find the optimal weights.
    learned_weights = np.array([0.1, -0.2, 0.3, -0.05, 0.15])
    learned_logloss = log_loss_accuracy_obj.get_logloss(learned_weights)
    print(f"\nLog Loss on the test set with (imagined) learned weights: {learned_logloss:.4f}")
    learned_accuracy = log_loss_accuracy_obj.get_accuracy(learned_weights)
    print(f"Accuracy on the test set with (imagined) learned weights: {learned_accuracy:.4f}")

    # Demonstrate the classify function
    first_test_sample = X_test[0]
    prediction = log_loss_accuracy_obj.classify(learned_weights, threshold=0.5)
    print(f"\nPredictions for the test set (first 5 samples): {prediction[:5]}")
    print(f"True labels for the test set (first 5 samples): {y_test[:5]}")