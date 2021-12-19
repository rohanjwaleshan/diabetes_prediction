# Diagnosing Patients with Type 2 Diabetes: Project Overview
* Trained a classifer to predict whether a patient has Type 2 diabetes or not (f1-score = 0.67, recall = 0.83, precision = 0.56 \*\*positive class\*\*), so doctors can have assistance in diagnosing patients
* The dataset used to train the model was accessed from [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database). All patients in the dataset are females, at least 21 years old, and of Pima Indian heritage.
* 768 patients (before removing observations from dataset)
* Identified missing values that were recorded as zeros and dropped/imputed them
* Used SMOTE to handle class imbalance (66% negative cases, 34% positive cases)
* Created a pairplot with response as category to see if data was linealry separable
* Created a pipeline with imblearn library for necessary preprocessing steps to be performed during GridSearchCV and RandomizedSearchCV
* Recall was primarily considered for evaluating random forest and non-linear svm models (test set was imbalanced and minimizing false negatives was important)

## Code
* **Python Version**: 3.8.8
* **Packages**: numpy, pandas, matplotlib, seaborn, sklearn, imblearn, pickle, flask
* **Requirements**: `pip install -r requirements.txt`

## Data
The dataset used for training the classifiers was accessed from [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database) but originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The dataset contains information about 768 patients who were all female, 21 years or older, and of Pima Indian heritage.
* **Pregnancies** - number of pregnancies
* **Glucose** - plasma glucose concentration after two hours in an oral glucose tolerance test
* **BloodPressure** - diastolic blood pressure (mm Hg)
* **SkinThickness** - triceps skin fold thickness (mm)
* **Insulin** - two hour serum insulin (muU/mL)
* **BMI** - body mass index (weight in kg/(height in m)^2)
* **DiabetesPedigreeFunction** - diabetes pedigree function
* **Age** - age in years
* **Outcome** - has diabetes or not (1 or 0, respectively)

## EDA
Performing descriptive statistics constructing histograms for each variable in the dataset led to discovering variables that had values of zero where it wasn't logically possible. (Insert image of zeros here). Section 3.7 of the following scientific [paper](https://www.sciencedirect.com/science/article/pii/S2352914816300016#s0050) explained how the zeros for these observations were actually recorded in place of missing values. Therefore I removed observations with very few "zeros" (Glucose, BloodPressure, BMI) and imputed the median (skewed distributions for SkinThickness and Insulin) for the rest before training the models.

I checked whether the data was balanced before training the models (to prevent getting high accuracy just by predicting the majority class):
(insert image of bar chart class imbalance)

Since the data was slightly imbalanced I decided to use SMOTE to balance training data before training models.

I created a pairplot with the response (Outcome) as a label for each scatterplot to determine if data was linearly separable:
(insert image of pairplot)

Since the data didn't seem to be linealry separable I proceeded with models that didn't require data to be linearly separable.

## Model Development
### Random Forest
* Split data into train and test sets
* Converted zeros in columns SkinThickness and Insulin to missing values (to be imputed with median in Pipeline)
* Created a Pipeline with essential steps (median imputer, SMOTE, and random forest classifer)
* Performed GridSearchCV with 5 fold cross validation (hyperparameters were n_estimators, max_features, and max_depth) to get best model based on recall
* Evaluated the best estimated model (based on GridSearchCV results) on unseen test set

### Non-linear SVM
* Split data into train and test sets
* Converted zeros in columns SkinThickness and Insulin to missing values (to be imputed with median in Pipeline)
* Created a Pipeline with essential steps (median imputer, min max scaler, SMOTE, and support vector classifer)
* Performed RandomizedSearchCV (due to complexity) with 5 fold cross validation (hyperparameters were kernel, C, degree, and gamma) to get best model based on recall
* Evaluated the best estimated model (based on RandomizedSearchCV results) on unseen test set



