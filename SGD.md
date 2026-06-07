# SGD (Stochastic Gradient Descent)

## What is SGD?

Stochastic Gradient Descent (SGD) is an optimization algorithm used to train machine learning models by iteratively updating model parameters using small batches of data. It is especially useful for large datasets where computing the full gradient is expensive.

## How SGD Works

1. Initialize model weights randomly or with zeros.
2. Select a single sample or mini-batch from the training set.
3. Compute the gradient of the loss with respect to the current weights.
4. Update the weights by moving in the opposite direction of the gradient.
5. Repeat until convergence or until a fixed number of iterations.

This process can be summarized as:

- `w = w - learning_rate * gradient`

Where:

- `w` is the parameter vector.
- `learning_rate` controls the step size.
- `gradient` is the derivative of the loss function.


<img src="https://miro.medium.com/v2/0*c_MqktGZe2_2Gb0u.jpg" alt="SGD Image">


## Why Use SGD?

- Efficient for large datasets
- Works well with online and incremental learning
- Commonly used in linear models, neural networks, and regression

## Project Structure

- `SGD_Regressor.py` - main regression workflow for training an SGD-based model.
- `MAE.py` - mean absolute error calculation for model evaluation.
- `requirements.txt` - dependency list.
- `LICENSE` - license file.

## Usage Notes

1. Install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Add your dataset loading and preprocessing to `SGD_Regressor.py`.
3. Train the model using SGD and generate predictions.
4. Evaluate predictions using the mean absolute error helper in `MAE.py`.

## Gradient Descent vs Stochastic Gradient Descent

- Gradient Descent uses the entire dataset to compute each update.
- Stochastic Gradient Descent uses one sample or a small batch for each update.
- SGD is typically faster and more scalable for large datasets.

## Notes

This repository is a lightweight starting point for implementing an SGD regressor workflow. Customize `SGD_Regressor.py` with your own data, model configuration, and evaluation steps.
