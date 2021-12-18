# Diagnosing Patients with Type 2 Diabetes: Project Overview
* Trained a classifer to predict whether a patient has Type 2 diabetes or not (f1-score = 0.74, recall = 0.77, precision = 0.74 \*\*macro averages\*\*), so doctors can have assistance in diagnosing patients
* The dataset used to train the model was accessed from [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database). All patients in the dataset are females, at least 21 years old, and of Pima Indian heritage.
* 768 patients (before removing observations from dataset)
* Identified missing values that were recorded as zeros and dropped/imputed them
* Used SMOTE to handle class imbalance (66% negative cases, 34% positive cases)
* Created a pairplot with response as category to see if data was linealry separable
* Created a pipeline with imblearn library for necessary preprocessing steps to be performed during GridSearchCV and RandomizedSearchCV
* F-score and recall were primarily considered for evaluating random forest and non-linear svm models (test set was imbalanced)

## Code
* **Python Version**: 3.8.8
* **Packages**: numpy, pandas, matplotlib, seaborn, sklearn, imblearn, pickle, flask
* **Requirements**: `pip install -r requirements.txt`

## Data
The dataset used for training the classifiers was accessed from [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database) but originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The dataset contained information about 768 patients who were all female, 21 years or older, and of Pima Indian heritage.
* **Pregnancies** - number of pregnancies
* **Glucose** - plasma glucose concentration after two hours in an oral glucose tolerance test
* **BloodPressure** - diastolic blood pressure (mm Hg)
* **SkinThickness** - triceps skin fold thickness (mm)
* **Insulin** - two hour serum insulin ($\mu$U/mL)
* **BMI** - body mass index (weight in kg/$(height in m)^{2}$)
* **DiabetesPedigreeFunction** - diabetes pedigree function
* **Age** - age in years
* **Outcome** - has diabetes or not (1 or 0, respectively)
