# Linear Regression From Scratch

This note connects the math to the implementation in `src/linear_regression_from_scratch.py`.

## 1. Dataset

We start with a tiny supervised learning dataset:

```text
x: 1, 2, 3, 4, 5
y: 3, 5, 7, 9, 11
```

The true hidden pattern is:

```text
y = 2x + 1
```

But the model does not know that. It must learn the weight and bias from errors.

## 2. Model

The model is:

```text
y_hat = wx + b
```

At the beginning we choose:

```text
w = 0
b = 0
```

So every prediction starts as:

```text
y_hat = 0x + 0 = 0
```

That is a bad model, but it gives us errors. Those errors are exactly what learning needs.

## 3. Error

For one data point:

```text
error = actual - predicted
error = y - y_hat
```

If `x = 1` and `y = 3`, the initial model predicts:

```text
y_hat = 0
error = 3 - 0 = 3
```

The model is underpredicting.

## 4. Mean Squared Error

We square each error and average:

```text
MSE = (1 / n) * sum((y_i - y_hat_i)^2)
```

We square errors because:

- positive and negative errors should not cancel
- large errors should be punished more
- the function becomes smooth and differentiable
- for linear regression, the loss surface is convex

## 5. Gradients

The gradient tells us the direction and steepness of the loss surface.

For the weight:

```text
dMSE/dw = -(2 / n) * sum((y_i - y_hat_i) * x_i)
```

For the bias:

```text
dMSE/db = -(2 / n) * sum(y_i - y_hat_i)
```

The important intuition is:

```text
gradient for w depends on error * feature
gradient for b depends on error
```

The weight controls how strongly the feature affects prediction. So its gradient must care about both the error and the feature value.

The bias shifts all predictions up or down. So its gradient only needs the total error direction.

## 6. Gradient Descent

To reduce loss, move parameters opposite the gradient:

```text
w = w - learning_rate * dMSE/dw
b = b - learning_rate * dMSE/db
```

At epoch 0 in our script:

```text
w = 0
b = 0
loss = 57
dMSE/dw = -50
dMSE/db = -14
```

With learning rate `0.01`:

```text
w = 0 - 0.01 * (-50) = 0.5
b = 0 - 0.01 * (-14) = 0.14
```

Because the gradients are negative, subtracting them increases `w` and `b`. That makes sense because the model was predicting values that were too small.

## 7. What Convergence Means

After training, the script learns approximately:

```text
y_hat = 2.0039x + 0.9860
```

The true relationship is:

```text
y = 2x + 1
```

So gradient descent discovered the line from data.

## 8. Practice Experiments

Try these one at a time:

1. Change `learning_rate` from `0.01` to `0.001`.
   Notice that learning becomes slower.

2. Change `learning_rate` from `0.01` to `0.1`.
   Watch whether learning is faster or unstable.

3. Change the initial values of `w` and `b`.
   Confirm that a convex loss still moves toward the same solution.

4. Change the dataset to:

   ```text
   xs = [1, 2, 3, 4, 5]
   ys = [5, 7, 9, 11, 13]
   ```

   Predict what `w` and `b` should become before running the script.

5. Add noise:

   ```text
   ys = [3.2, 4.8, 7.3, 8.7, 11.1]
   ```

   Notice that the model will find the best line, not a perfect line.

## 9. Connection to Predictive Maintenance

In the final project, the features will be sensor readings instead of one `x` value:

```text
temperature
vibration
pressure
rotational_speed
tool_wear
```

The same idea remains:

```text
data -> prediction -> error -> gradient -> update
```

Linear regression predicts continuous values. Later, logistic regression will use the same learning pattern to predict probabilities, such as:

```text
probability of machine failure = 0.82
```

That probability will become the foundation for risk scoring and human escalation.

