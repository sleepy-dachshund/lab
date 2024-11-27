import numpy as np


def add_regularization(cost: float, w: np.array, reg_type: str = 'ridge', lambda_: float = 0.1):
    """
    Add regularization term to any cost function.

    Args:
        cost (float): Original cost value
        w (np.array): Weight parameters
        reg_type (str): 'ridge' or 'lasso'
        lambda_ (float): Regularization strength

    Returns:
        float: Cost with regularization term added
    """
    if reg_type in ['ridge', 'l2']:
        return cost + 0.5 * lambda_ * np.sum(w ** 2)
    elif reg_type in ['lasso', 'l1']:
        return cost + lambda_ * np.sum(np.abs(w))
    return cost


def regularization_gradient(w: np.array, reg_type: str = 'ridge', lambda_: float = 0.1):
    """
    Calculate gradient of regularization term.

    Args:
        w (np.array): Weight parameters
        reg_type (str): 'ridge' or 'lasso'
        lambda_ (float): Regularization strength

    Returns:
        np.array: Gradient of regularization term
    """
    if reg_type in ['ridge', 'l2']:
        return lambda_ * w
    elif reg_type in ['lasso', 'l1']:
        return lambda_ * np.sign(w)
    return 0


def mse_cost(X: np.array, y: np.array, w: np.array, b: float,
             reg_type: str = None, lambda_: float = 0.1):
    """
    Mean Squared Error cost function with optional regularization.

    Args:
        X (np.array): Feature matrix
        y (np.array): Target values
        w (np.array): Weights
        b (float): Bias term
        reg_type (str): Type of regularization ('ridge', 'lasso', or None)
        lambda_ (float): Regularization strength

    Returns:
        float: Cost value
    """
    y_hat = X.dot(w) + b
    cost = 0.5 * np.mean((y - y_hat) ** 2)
    return add_regularization(cost, w, reg_type, lambda_)


def mae_cost(X: np.array, y: np.array, w: np.array, b: float,
             reg_type: str = None, lambda_: float = 0.1):
    """
    Mean Absolute Error cost function with optional regularization.
    Better for outliers than MSE.
    """
    y_hat = X.dot(w) + b
    cost = np.mean(np.abs(y - y_hat))
    return add_regularization(cost, w, reg_type, lambda_)


def huber_cost(X: np.array, y: np.array, w: np.array, b: float,
               delta: float = None, reg_type: str = None, lambda_: float = 0.1):
    """
    Huber loss: quadratic for small errors, linear for large ones.
    Combines MSE and MAE.
    """
    y_hat = X.dot(w) + b
    errors = y - y_hat
    if delta is None:
        delta = np.quantile(np.abs(errors), 0.9)
    cost = np.mean(np.where(np.abs(errors) <= delta,
                            0.5 * errors ** 2,
                            delta * np.abs(errors) - 0.5 * delta ** 2))
    return add_regularization(cost, w, reg_type, lambda_)


def mape_cost(X: np.array, y: np.array, w: np.array, b: float,
              epsilon: float = 1e-8, reg_type: str = None, lambda_: float = 0.1):
    """
    Mean Absolute Percentage Error with optional regularization.
    Good for relative errors, scale-independent.
    """
    y_hat = X.dot(w) + b

    # Calculate percentage errors avoiding division by zero
    denominators = np.maximum(np.abs(y), epsilon)
    percentage_errors = np.abs((y - y_hat) / denominators)

    # Calculate mean and convert to percentage
    cost = np.mean(percentage_errors) * 100
    return add_regularization(cost, w, reg_type, lambda_)


def msle_cost(X: np.array, y: np.array, w: np.array, b: float,
              reg_type: str = None, lambda_: float = 0.1):
    """
    Mean Squared Logarithmic Error with optional regularization.
    Good for right-skewed distributions and when target spans multiple scales.
    """
    y_hat = X.dot(w) + b

    # Handle zero/negative values by shifting
    y_shift = y + np.abs(np.min(y)) + 1.0 if np.min(y) <= 0 else y
    y_hat_shift = y_hat + np.abs(np.min(y)) + 1.0 if np.min(y) <= 0 else y_hat

    cost = np.mean((np.log1p(y_shift) - np.log1p(y_hat_shift)) ** 2)
    return add_regularization(cost, w, reg_type, lambda_)


def gradient_factory(cost_func):
    """
    Factory function to create gradient functions using numerical differentiation.
    Useful for complex cost functions where analytical gradients are hard to derive.

    Args:
        cost_func: Cost function to differentiate

    Returns:
        tuple: (dw_func, db_func) gradient functions for weights and bias
    """
    def dw_cost(X: np.array, y: np.array, w: np.array, b: float,
                reg_type: str = None, lambda_: float = 0.1):
        epsilon = 1e-8
        n_features = w.shape[0]
        gradient = np.zeros_like(w)

        for i in range(n_features):
            w_plus = w.copy()
            w_plus[i] += epsilon
            w_minus = w.copy()
            w_minus[i] -= epsilon

            gradient[i] = (cost_func(X=X, y=y, w=w_plus, b=b, reg_type=reg_type, lambda_=lambda_) -
                           cost_func(X=X, y=y, w=w_minus, b=b, reg_type=reg_type, lambda_=lambda_)) / (2 * epsilon)

        if reg_type:
            gradient += regularization_gradient(w, reg_type, lambda_)
        return gradient

    def db_cost(X: np.array, y: np.array, w: np.array, b: float,
                reg_type: str = None, lambda_: float = 0.1):
        epsilon = 1e-8
        return (cost_func(X=X, y=y, w=w, b=b + epsilon, reg_type=reg_type, lambda_=lambda_) -
                cost_func(X=X, y=y, w=w, b=b - epsilon, reg_type=reg_type, lambda_=lambda_)) / (2 * epsilon)

    return dw_cost, db_cost


def gradient_descent_generic(X: np.array, y: np.array,
                             cost_func, dw_func, db_func,
                             reg_type: str = None, lambda_: float = 0.1,
                             learning_rate: float = 0.02, num_iterations: int = 100):
    """
    Generic gradient descent optimizer that works with any differentiable cost function.

    Args:
        X (np.array): Feature matrix
        y (np.array): Target values
        cost_func (function): Cost function to optimize
        dw_func (function): Weight gradient function
        db_func (function): Bias gradient function
        reg_type (str): Type of regularization ('ridge', 'lasso', or None)
        lambda_ (float): Regularization strength
        learning_rate (float): Learning rate for gradient descent
        num_iterations (int): Number of iterations

    Returns:
        tuple: (w, b, cost_history) Final weights, bias, and cost history
    """
    w = np.zeros(X.shape[1])
    b = 0
    cost_history = []

    for _ in range(num_iterations):
        dw = dw_func(X=X, y=y, w=w, b=b, reg_type=reg_type, lambda_=lambda_)
        db = db_func(X=X, y=y, w=w, b=b, reg_type=reg_type, lambda_=lambda_)

        w -= learning_rate * dw
        b -= learning_rate * db

        cost = cost_func(X=X, y=y, w=w, b=b, reg_type=reg_type, lambda_=lambda_)
        cost_history.append(cost)

    return w, b, cost_history


if __name__ == '__main__':
    from utils_linear_regression import LinearRegressionDataGenerator

    generator = LinearRegressionDataGenerator(n_samples=5000,
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

    # Create gradient functions for each cost function
    dw_mse, db_mse = gradient_factory(mse_cost)
    dw_mae, db_mae = gradient_factory(mae_cost)
    dw_huber, db_huber = gradient_factory(huber_cost)
    dw_mape, db_mape = gradient_factory(mape_cost)
    dw_msle, db_msle = gradient_factory(msle_cost)

    iterations = 10000

    # Run gradient descent for each cost function
    w_mse, b_mse, cost_history_mse = gradient_descent_generic(X, y, mse_cost, dw_mse, db_mse, num_iterations=iterations)
    w_mae, b_mae, cost_history_mae = gradient_descent_generic(X, y, mae_cost, dw_mae, db_mae, num_iterations=iterations)
    w_huber, b_huber, cost_history_huber = gradient_descent_generic(X, y, huber_cost, dw_huber, db_huber, num_iterations=iterations)
    # w_mape, b_mape, cost_history_mape = gradient_descent_generic(X, y, mape_cost, dw_mape, db_mape, num_iterations=iterations)
    w_msle, b_msle, cost_history_msle = gradient_descent_generic(X, y, msle_cost, dw_msle, db_msle, num_iterations=iterations)

    # Print results
    print(f'True Weights: {generator.true_coef}')
    print(f'Weights (MSE): {w_mse}')
    print(f'Weights (MAE): {w_mae}')
    print(f'Weights (Huber): {w_huber}')
    # print(f'Weights (MAPE): {w_mape}')
    print(f'Weights (MSLE): {w_msle}')

    # Plot cost histories
    import matplotlib.pyplot as plt
    plt.plot(cost_history_mse, label='MSE')
    plt.plot(cost_history_mae, label='MAE')
    plt.plot(cost_history_huber, label='Huber')
    # plt.plot(cost_history_mape, label='MAPE')
    plt.plot(cost_history_msle, label='MSLE')
    plt.legend()
    plt.show()

    # Plot predictions vs. true values
    y_hat_huber = X.dot(w_huber) + b_huber
    plt.scatter(y, y_hat_huber)
    plt.axvline(np.mean(y), color='red', linestyle='--')
    plt.plot([np.min(y), np.max(y)], [np.min(y), np.max(y)], color='green')
    plt.xlabel('True Target')
    plt.ylabel('Predicted Target')
    plt.title('Huber Cost')
    plt.show()
