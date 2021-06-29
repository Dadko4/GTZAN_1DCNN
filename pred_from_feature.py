import numpy as np
from pandas import read_csv
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = read_csv('features_3_sec.csv')
data = df.values
X = data[:, 1:-1]
y = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)
preds = model.predict(X_test)
accuracy_score(y_test, preds)
