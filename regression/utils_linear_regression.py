
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KernelDensity
from pandas.tseries.offsets import BDay


class LinearRegressionDataGenerator:
    def __init__(self,
                 n_samples: int, n_features: int,
                 y_mean: float, y_std: float,
                 true_coef: list,
                 noise_mean: float, noise_std: float,
                 start_date: str,
                 y_input: np.array = None,
                 random_seed: int = 252):

        # option to pass in y_input for reproducibility
        self.y_input = y_input
        if self.y_input is not None:
            n_samples = len(y_input)
            y_mean = np.mean(y_input)
            y_std = np.std(y_input)

        self.n_samples = n_samples
        self.n_features = n_features
        self.features = None
        self.y_mean = y_mean
        self.y_std = y_std
        self.true_coef = np.array(true_coef)
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.start_date = pd.Timestamp(start_date)

        self.estimated_coef = None
        self.estimated_intercept = None
        self.theta = None

        # Set random seed for reproducibility
        np.random.seed(random_seed)

        self._generate_data()
        self._create_dataframe()
        self._fit_linear_model()

    def _generate_data(self):
        if self.y_input is not None:
            self.y = self.y_input
        else:
            # Generate random data for target variable y
            self.y = np.random.normal(self.y_mean, self.y_std, size=self.n_samples)

        # Generate random values for n_features - 1 (all but the last feature, which will be the plug)
        X_random = np.random.rand(self.n_samples, self.n_features - 1)

        # Generate random noise
        noise = np.random.normal(self.noise_mean, self.noise_std, size=self.n_samples)

        # Calculate the part of y explained by the random features and noise
        y_partial = X_random.dot(self.true_coef[:-1]) + noise

        # Solve for the final feature to satisfy the equation y = X * true_coef + noise
        X_remaining = (self.y - y_partial) / self.true_coef[-1]

        # Stack the random features with the calculated feature to create the full X matrix
        self.X = np.hstack((X_random, X_remaining.reshape(-1, 1)))

    def _create_dataframe(self):
        # Generate business day index for the specified number of samples
        date_index = pd.bdate_range(start=self.start_date.strftime('%Y-%m-%d'), periods=self.n_samples)

        # Create the DataFrame with explanatory data (X) and target data (y)
        self.df = pd.DataFrame(self.X, index=date_index, columns=[f"feature{ i +1:03}" for i in range(self.n_features)])
        self.df['target_shifted'] = self.y
        self.df['target'] = self.df['target_shifted'].shift(1)
        self.features = [col for col in self.df.columns if 'feature' in col]

    def _fit_linear_model(self):
        # Fit the linear regression model using scikit-learn
        model = LinearRegression()
        model.fit(self.X, self.y)
        self.estimated_coef = model.coef_
        self.estimated_intercept = model.intercept_
        self.theta = np.hstack((model.intercept_, model.coef_))

    def get_X(self):
        return self.X

    def get_y(self):
        return self.y

    def get_dataframe(self):
        return self.df


if __name__ == '__main__':
    generator = LinearRegressionDataGenerator(n_samples=100,
                                              n_features=3,
                                              y_mean=1,
                                              y_std=2,
                                              true_coef=[1.5, -2.0, 0.5],
                                              noise_mean=0,
                                              noise_std=0.1,
                                              start_date='2023-01-01')
    X = generator.get_X()
    y = generator.get_y()
    df = generator.get_dataframe()

    # Fit a linear model
    print("True Weights:", generator.true_coef)
    print("Estimated Weights:", generator.estimated_coef.round(4))
    print("Intercept:", round(generator.estimated_intercept, 4))
