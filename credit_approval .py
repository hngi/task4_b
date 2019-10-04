import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Read the data file
CC_DATA = pd.read_csv("CC_data.csv")
# Replace "?" with NaN
CC_DATA.replace('?', np.NaN, inplace=True)
# Convert Age to numeric
CC_DATA["Age"] = pd.to_numeric(CC_DATA["Age"])
# CC_data2 = CC_DATA[:,:]
CC_DATA2 = CC_DATA.copy()

# view dataset
CC_DATA.describe()

# count missing values
CC_DATA.isnull().sum()

# Inputing missing values for numerical columns with mean value
CC_DATA.fillna(CC_DATA.mean(), inplace=True)

# Iterate over each column of CC_DATA
CC_DATA = CC_DATA.fillna(method='ffill')
# Count the number of NaNs in the dataset and print the counts to verify
print(CC_DATA.count())
CC_DATA.tail(20)

#print dataset info
CC_DATA.info()

#plot graphs for analysis
def plot_dist_plot(col):
    """Flexibly plot a univariate distribution of observation"""
    sns.distplot(col)
    plt.show()
plot_dist_plot(CC_DATA['Age'])
plot_dist_plot(CC_DATA['Debt'])
plot_dist_plot(CC_DATA['YearsEmployed'])
plot_dist_plot(CC_DATA['CreditScore'])
plot_dist_plot(CC_DATA['Income'])

#correlation matrix
CORRMAT = CC_DATA.corr()
F, AX = plt.subplots(figsize=(12, 9))
sns.heatmap(CORRMAT, vmax=.8, square=True)

#scatterplot
sns.set()
COLS = ['Age', 'Income', 'CreditScore', 'Debt', 'YearsEmployed']
sns.pairplot(CC_DATA[COLS], size=2.5)
plt.show()

# check for approved
sns.countplot(data=CC_DATA, x='Approved')

#recount null values
CC_DATA.isnull().sum()

print("shape of the data:", CC_DATA.shape)

#converting non-numeric to numeric values
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
# # Looping for each object type column
#Using label encoder to convert into numeric types
for col in CC_DATA:
    if CC_DATA[col].dtypes == 'object':
        CC_DATA[col] = LE.fit_transform(CC_DATA[col])

from sklearn.model_selection import train_test_split
# Removing the feature which are not important and converting to NumPy array
CC_DATA = CC_DATA.drop(['DriversLicense', 'ZipCode'], axis=1)
CC_DATA = CC_DATA.values

# Creating new variable to input features and labels
X, Y = CC_DATA[:, 0:13], CC_DATA[:, 13]

# Spliting the data into training and testing sets
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.2, random_state=123)

# Import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# Rescaling X_train and X_test
SCALER = MinMaxScaler(feature_range=(0, 1))
RESCALERX_TRAIN = SCALER.fit_transform(X_TRAIN)
RESCALERX_TEST = SCALER.transform(X_TEST)

RESCALERX = SCALER.transform(X)

#Using RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
RF = RandomForestClassifier(n_estimators=500)
RF.fit(RESCALERX_TRAIN, Y_TRAIN)
Y_PRED = RF.predict(RESCALERX_TEST)
print("Random Forest classifier has accuracy of: ", RF.score(RESCALERX_TEST, Y_TEST))
# Evaluate the confusion_matrix
confusion_matrix(Y_TEST, Y_PRED)

IMPORTANCES = RF.feature_importances_
STD = np.std([tree.feature_importances_ for tree in RF.estimators_],
             axis=0)
INDICIES = np.argsort(IMPORTANCES)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, INDICIES[f], IMPORTANCES[INDICIES[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), IMPORTANCES[INDICIES], color="r", yerr=STD[INDICIES], align="center")
plt.xticks(range(X.shape[1]), INDICIES)
plt.xlim([-1, X.shape[1]])
plt.show()

# copy of CC_data is in CC_DATA2
CC_DATA2 = CC_DATA2.drop(['Approved'], axis=1)

FEATURES = CC_DATA2.columns
IMPORTANCE = RF.feature_importances_
INDICIES = np.argsort(IMPORTANCES)

plt.title('Feature Importances')
plt.barh(range(len(INDICIES)), IMPORTANCES[INDICIES], color='b', align='center')
plt.yticks(range(len(INDICIES)), [FEATURES[i] for i in INDICIES])
plt.xlabel('Relative Importance')
plt.show()

# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Fitting logistic regression with default parameter values
LOGRED = LogisticRegression()
LOGRED.fit(RESCALERX_TRAIN, Y_TRAIN)

# Import confusion_matrix
from sklearn.metrics import confusion_matrix

# Using the trained model to predict instances from the test set
Y_PRED = LOGRED.predict(RESCALERX_TEST)

# Getting the accuracy score of predictive model
print("Logistic regression classifier has accuracy of: ", LOGRED.score(RESCALERX_TEST, Y_TEST))

# Evaluate the confusion_matrix
confusion_matrix(Y_TEST, Y_PRED)

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Define the grid of values for tol and max_iter
TOL = [0.01, 0.001, 0.0001]
MAX_ITER = [100, 150, 200]

# Create a dictionary
PARAM_GRID = dict(tol=TOL, max_iter=MAX_ITER)

# Initializing GridSearchCV
GRID_MODEL = GridSearchCV(estimator=LogisticRegression(), param_grid=PARAM_GRID, cv=5)

# Rescaling the entire data set with all the predictive features
RESCALERX = SCALER.fit_transform(X)

# Calculating and summarizing the final results
GRID_MODEL_RESULT = GRID_MODEL.fit(RESCALERX, Y)
BEST_SCORE, BEST_PARAMS = GRID_MODEL_RESULT.best_score_, GRID_MODEL_RESULT.best_params_
print("Best: %f using %s" %  (BEST_SCORE, BEST_PARAMS))
