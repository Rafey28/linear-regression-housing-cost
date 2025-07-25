# linear-regression-housing-cost
A basic linear regression model from scratch in Python to predict housing prices.

## 📊 Dataset

| Size (1000 sqft) | Price ($1000s) |
|------------------|----------------|
| 1.0              | 100            |
| 2.0              | 150            |
| 3.0              | 200            |
| 4.0              | 250            |
| 5.0              | 300            |

## 📈 Model

The model is trained using gradient descent to minimize the cost function:
J(w, b) = (1/2m) * Σ((f_wb(x) - y)^2)

## 🔧 Features

- Manual cost function
- Manual gradient computation
- Gradient descent optimization
- Visualization using `matplotlib`

## 🔮 Prediction

Example:
> Predicted price for 6k sqft: **$349.92k**

## 🧠 Learnings

- Linear regression fundamentals
- Gradient descent algorithm
- Numpy operations
- Data visualization with matplotlib

## 🚀 Run it

```bash
python housing_price_model.py
