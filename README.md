# 🏠 California Housing Price Prediction using Deep Neural Networks (Keras)
📁 **Portfolio Project by Aishwarya**

This end-to-end machine learning project demonstrates how to predict housing prices in California using a deep neural network built with TensorFlow/Keras. The implementation showcases modern deep learning techniques such as dropout regularization, L2 regularization, and early stopping. The project also emphasizes model tuning and performance monitoring on real-world regression data.

You can explore the notebook and run it live using Google Colab:  
[Open Colab Notebook](https://github.com/Aishwaryachen11/California_Housing_price_Predictor/blob/main/California_Housing_Dataset.ipynb)

---

## 📌 Project Overview

- **Goal**: Predict the median housing prices in various California districts.
- **Dataset**: California Housing Dataset (available via `sklearn.datasets`)
- **Tools Used**: Python, TensorFlow, Keras, Pandas, NumPy, Matplotlib, Scikit-learn
- **Problem Type**: Supervised Regression

---

## 🧪 Dataset Information

- **Source**: [California Housing dataset](https://keras.io/api/datasets/california_housing/)
- **Features**: 8 numerical input features (e.g., Median Income, Avg. Rooms, Population)
- **Target**: Median house value (in 100,000s)
- **Total Samples**: ~20,000
- **Preprocessing**:
  - Data normalization using `StandardScaler`
  - Train/validation/test split (60/20/20)

---

## 🏗️ Model Architecture

A **custom feedforward deep neural network** was designed for regression using `Keras.Sequential()`:

| Layer Type | Neurons | Activation | Regularization |
|------------|---------|------------|----------------|
| Dense      | 128     | ReLU       | L2 (0.01)      |
| Dropout    | —       | —          | 0.3 dropout    |
| Dense      | 64      | ReLU       | L2 (0.01)      |
| Dense      | 32      | ReLU       | —              |
| Output     | 1       | Linear     | —              |

- **Optimizer**: Adam (`learning_rate=0.001`)
- **Loss Function**: Mean Squared Error (MSE)
- **Epochs**: Up to 50 with early stopping

---

## 🧠 Key Techniques Implemented

✅ Custom deep learning model for regression  
✅ Dropout Regularization to prevent overfitting  
✅ L2 Regularization to penalize large weights  
✅ Adam optimizer with a tuned learning rate  
✅ EarlyStopping callback (patience=5) to restore best model  
✅ Training and validation loss visualization  
✅ Final testing with sample predictions

---

## 📈 Learning Curve

We visualized the training and validation loss to monitor learning performance and detect overfitting.

<img src="https://github.com/Aishwaryachen11/California_Housing_Predictor/blob/main/Images/training_validation_loss.png" alt="Training and Validation Loss" width="600"/>

---

## 🧪 Evaluation on Test Data

After training, the model was evaluated on the test dataset:

- **Final Test MSE**: ~0.3621  
- Sample predictions:

| Sample | Predicted Value | Actual Value |
|--------|------------------|--------------|
| 1      | 0.73             | 0.48         |
| 2      | 1.68             | 0.46         |
| 3      | 4.26             | 5.00         |


