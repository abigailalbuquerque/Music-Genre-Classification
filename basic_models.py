import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import pickle

# Reading the data from csv into dataframe
data = pd.read_csv('data.csv')
print(data.head())

# Shuffling the data
data = data.sample(frac=1)
print(data.head())

# Separating our X and y
y = data['Genre']
X = data.loc[:, data.columns != 'Genre']
print(X.head())

# Normalising the dataset
scaler = StandardScaler()
X_normal = scaler.fit_transform(np.array(X, dtype=float))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_normal, y, test_size=0.2)

# Encoding the labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)


def model_assess(model, title="Default"):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds = le.inverse_transform(preds)
    print('Accuracy', title, ':', round(accuracy_score(y_test, preds), 5))


nb = GaussianNB()
model_assess(nb, "Naive Bayes")

sgd = SGDClassifier(max_iter=5000, random_state=0)
model_assess(sgd, "Stochastic Gradient Descent")

knn = KNeighborsClassifier(n_neighbors=12)
model_assess(knn, "KNN")

tree = DecisionTreeClassifier()
model_assess(tree, "Decision trees")

rforest = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
model_assess(rforest, "Random Forest")

lg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=7000)
model_assess(lg, "Logistic Regression")

svm = SVC(decision_function_shape="ovo",probability=True)
model_assess(svm, "Support Vector Machine Proba")

nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5000, 10), random_state=1)
model_assess(nn, "Neural Nets")

ada = AdaBoostClassifier(n_estimators=1000)
model_assess(ada, "AdaBoost")

# Cross Gradient Booster
xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)
model_assess(xgb, "Cross Gradient Booster")

# Cross Gradient Booster (Random Forest)
xgbrf = XGBRFClassifier(objective='multi:softmax')
model_assess(xgbrf, "Cross Gradient Booster (Random Forest)")

