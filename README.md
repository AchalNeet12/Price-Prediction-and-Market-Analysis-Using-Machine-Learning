# Price Prediction and Market Analysis Using Machine Learning
---
---

## Overview:
This project focuses on analyzing the avocado market and building a machine learning model to predict the price of avocados based on various features such as region, year, volume, and avocado type. The goal is to answer critical business questions about avocado pricing and production patterns,  Several machine learning models were trained, and XGBoost showed the highest accuracy in price prediction. An interactive web app was developed using Streamlit, allowing users to predict avocado prices based on input variables, offering valuable insights into market trends and dynamics.
- **Business Questions Addressed:**
    - Which regions have the lowest and highest avocado prices?
    - What region has the highest avocado production?
    - What are the average avocado prices per year?
    - What are the average avocado volumes per year?
---

## Dataset Overview:
The dataset utilized in this project contains comprehensive data on avocado sales across different regions and years. It encompasses both numerical and categorical features that provide insights into market trends, pricing patterns, and production volumes.
   - **Date:** The date of the observation or transaction.
   - **AveragePrice:** The average price of a single avocado for the given observation.
   - **Total Volume:** The total number of avocados sold, measured by volume.
   - **Total Bags:** The total number of bags sold, including small, large, and extra-large bags.
   - **Small Bags:** The number of small-sized bags sold.
   - **Large Bags:** The number of large-sized bags sold.
   - **XLarge Bags:** The number of extra-large-sized bags sold.
   - **Type:** Indicates the type of avocado, with values "conventional" (0) and "organic" (1).
   - **Year:** The year of the observation.
   - **Region:** The specific region or city where the data was recorded.
   - **4046:** The number of avocados sold with PLU code 4046.
   - **4225:** The number of avocados sold with PLU code 4225.
   - **4770:** The number of avocados sold with PLU code 4770.
---

## Key Features:
   - The dataset integrates both numerical features (such as total volume, average price, and bag sizes) and categorical features (such as avocado type and region).
   - The data spans several years, providing valuable temporal insights into pricing and sales patterns across different regions.
---

## Data Preprocessing:
   1. **Loading the Dataset:** The dataset is loaded into a Pandas DataFrame for analysis.
   2. **Handling Missing Values:** Missing data is checked and handled by removing or imputing values.
   3. **Encoding Categorical Variables:** Categorical columns (like 'Type' and 'Region') are encoded using numerical values and one-hot encoding.
   4. **Feature and Target Separation:** The target variable 'AveragePrice' is separated from the features (independent variables).
   5. **Splitting the Data:** The dataset is split into training and test sets (usually 70% for training and 30% for testing).
   6. **Feature Scaling:** Numerical features are standardized to have a mean of 0 and standard deviation of 1 to avoid scale dominance.
---

## Technologies Used:
   - **Python:** The core programming language used for data analysis, preprocessing, and model development.
   - **Pandas:** A powerful data manipulation library for reading, cleaning, and transforming the dataset.
   - **NumPy:** A library for numerical computations, used for array operations and mathematical functions.
   - **Scikit-learn:** A machine learning library for building regression models (e.g., Linear Regression, Random Forest, XGBoost), data splitting, and evaluation metrics.
   - **TensorFlow/Keras:** Used for building and training deep learning models (e.g., neural networks).
   - **Streamlit:** A framework for building interactive web applications to deploy the machine learning models and visualize results.
   - **Matplotlib/Seaborn:** Visualization libraries used for plotting graphs to analyze data trends and model performance.
---

## Model Training and Evaluation:
1. Model Selection: Multiple machine learning models are used to predict the target variable, AveragePrice. These models include:
    - `Linear Regression`
    - `Decision Tree Regressor`
    - `Random Forest Regressor`
    - `Support Vector Regressor (SVR)`
    - `K-Nearest Neighbors (KNN)`
    - `XGBoost Regressor`
    - `Deep Neural Network (DNN)`
      
 2. Model Training:
    - The dataset is split into training and test sets (typically 70% for training and 30% for testing).
    - For each model, the training set is used to fit the model, i.e., the model learns the relationship between the features and the target variable (AveragePrice).
      
 3. Model Evaluation:
  - After training, the models are evaluated on the test set using performance metrics such as:
    - Mean Absolute Error (MAE): Measures the average magnitude of errors between predicted and actual values.
    - Mean Squared Error (MSE): Penalizes larger errors more heavily than MAE, providing a better measure of model accuracy.
    - R-Squared (R²): Measures the proportion of variance in the target variable that is predictable from the input features.
    
 4. Results Comparison:
  - The models are compared based on their MAE, MSE, and R² scores. The model with the highest R² and lowest MAE and MSE is selected as the best performing model.

 5. Final Model Selection:
  - Based on the evaluation, XGBoost Regressor tends to show the best performance in terms of accuracy. Therefore, this model is used for deployment.

 6. Model Tuning (Optional):
  - Hyperparameters of the selected models, especially XGBoost, may be tuned using grid search or randomized search to improve accuracy further.
---

## Results:
  After training and evaluating multiple models, the following results were observed for each model based on Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-Squared (R²) scores:

  |      Model              |      MAE      |     MSE       |      R²       | 
  |-------------------------|---------------|---------------|---------------|
  | XGBoost                 |    0.094000   |    0.017000    |	0.900000    |
  | Random Forest           |    0.097000	|    0.019000	 |  0.886000    |               
  | K-nearest Neighbors     |    0.101000	|    0.025000	 |  0.852000    |              
  | Deep Neural Network     |    0.120795	|    0.027021	 |  0.837290    |             
  | Support Vector Machines |    0.118000	|    0.029000	 |  0.828000    |              
  | Decision Tree           |    0.137000	|    0.042000	 |  0.745000    |           
  | Linear Regression       |    0.184000	|    0.059000	 |  0.642000    |              

From the above table, XGBoost performs the best among all models, achieving the lowest MAE and MSE, and the highest R² score. This indicates that the XGBoost model has the best predictive accuracy and is the most reliable model for predicting the average price of avocados.

## Conclusion:
The XGBoost Regressor model outperformed all other models, including Linear Regression, Decision Tree, and Random Forest, with the lowest MAE, MSE, and the highest R² score. This indicates it provides the most accurate and reliable predictions for avocado prices. As a result, XGBoost was selected for deployment in the final application, offering a robust solution for real-time price predictions.
















