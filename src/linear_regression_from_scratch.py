"""
Linear Regression from Scratch

Goal:
    Understand supervised learning by implementing every major step manually.

Model:
    y_hat = w*x + b

Loss:
    MSE = (1/n) * sum((y_i - y_hat_i)^2)

Gradients:
    dMSE/dw = -(2/n) * sum((y_i - y_hat_i) * x_i)
    dMSE/db = -(2/n) * sum(y_i - y_hat_i)

Update:
    w = w - learning_rate * dMSE/dw
    b = b - learning_rate * dMSE/db
"""


def predict_one(x, w, b):
    """Predict one target value from one feature value."""
    return w * x + b


def predict_all(xs, w, b):
    """Predict target values for every x in the dataset."""
    predictions = []

    for x in xs:
        y_hat = predict_one(x, w, b)
        predictions.append(y_hat)

    return predictions


def compute_errors(ys, predictions):
    """Compute residuals: error = actual - predicted."""
    errors = []

    for y, y_hat in zip(ys, predictions):
        error = y - y_hat
        errors.append(error)

    return errors


def mean_squared_error(ys, predictions):
    """Compute average squared error."""
    total_squared_error = 0
    n = len(ys)

    for y, y_hat in zip(ys, predictions):
        error = y - y_hat
        squared_error = error ** 2
        total_squared_error += squared_error

    return total_squared_error / n


def compute_gradients(xs, ys, w, b):
    """
    Compute the slope of the loss surface at the current w and b.

    If a gradient is positive, increasing that parameter increases the loss.
    If a gradient is negative, increasing that parameter decreases the loss.
    """
    n = len(xs)
    predictions = predict_all(xs, w, b)
    errors = compute_errors(ys, predictions)

    error_feature_sum = 0
    error_sum = 0

    for x, error in zip(xs, errors):
        error_feature_sum += error * x
        error_sum += error

    gradient_w = -(2 / n) * error_feature_sum
    gradient_b = -(2 / n) * error_sum

    return gradient_w, gradient_b


def train_linear_regression(xs, ys, learning_rate, epochs):
    """Learn w and b using gradient descent."""
    w = 0
    b = 0
    history = []

    for epoch in range(epochs + 1):
        predictions = predict_all(xs, w, b)
        loss = mean_squared_error(ys, predictions)
        gradient_w, gradient_b = compute_gradients(xs, ys, w, b)

        history.append(
            {
                "epoch": epoch,
                "w": w,
                "b": b,
                "loss": loss,
                "gradient_w": gradient_w,
                "gradient_b": gradient_b,
            }
        )

        w = w - learning_rate * gradient_w
        b = b - learning_rate * gradient_b

    return w, b, history


def print_dataset(xs, ys):
    print("Dataset")
    print("x -> y")
    for x, y in zip(xs, ys):
        print(f"{x} -> {y}")
    print()


def print_training_progress(history, print_every):
    print("Training Progress")
    print("epoch | w        | b        | loss     | grad_w   | grad_b")
    print("-" * 65)

    for row in history:
        is_first = row["epoch"] == 0
        is_last = row["epoch"] == history[-1]["epoch"]
        should_print = row["epoch"] % print_every == 0

        if is_first or should_print or is_last:
            print(
                f"{row['epoch']:>5} | "
                f"{row['w']:>8.4f} | "
                f"{row['b']:>8.4f} | "
                f"{row['loss']:>8.4f} | "
                f"{row['gradient_w']:>8.4f} | "
                f"{row['gradient_b']:>8.4f}"
            )

    print()


def print_final_predictions(xs, ys, w, b):
    print("Final Model")
    print(f"y_hat = {w:.4f}x + {b:.4f}")
    print()

    print("Final Predictions")
    print("x | actual y | predicted y | error")
    print("-" * 40)

    predictions = predict_all(xs, w, b)
    errors = compute_errors(ys, predictions)

    for x, y, y_hat, error in zip(xs, ys, predictions, errors):
        print(f"{x} | {y:>8.4f} | {y_hat:>11.4f} | {error:>7.4f}")


def main():
    # A tiny dataset with an exact underlying pattern: y = 2x + 1.
    xs = [1, 2, 3, 4, 5]
    ys = [3, 5, 7, 9, 11]

    learning_rate = 0.01
    epochs = 1000

    print_dataset(xs, ys)

    final_w, final_b, history = train_linear_regression(
        xs=xs,
        ys=ys,
        learning_rate=learning_rate,
        epochs=epochs,
    )

    print_training_progress(history, print_every=100)
    print_final_predictions(xs, ys, final_w, final_b)


if __name__ == "__main__":
    main()
