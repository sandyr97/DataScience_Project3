#To Be, Or Not To Be
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
Load dataframe
df=pd.read_csv("../data/shakespeare_data.csv")
df.head(10)
df['PlayerLinenumber'] = df['PlayerLinenumber'].replace(np.nan, 0)
df['ActSceneLine'] = df['ActSceneLine'].replace(np.nan, "0.0.0")
df['Player'] = df['Player'].replace(np.nan, "NO ONE")
print(df.head(10))
print(df.shape)
print(df.describe())

temp = df["ActSceneLine"].str.split(".", n = 2, expand = True)
temp.head(10)
df["Act"]= temp[0]
df["Scene"]= temp[1]
df["Line"]= temp[1]
​
# Dropping old column
df.drop(columns =["ActSceneLine"], inplace = True)
print(df.head(10))

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['PlayLE'] = le.fit_transform(df["Play"].astype(str))
df['PlayerLE'] = le.fit_transform(df["Player"].astype(str))
df['LineTextLE'] = le.fit_transform(df["PlayerLine"].astype(str))
df['ActLE'] = le.fit_transform(df["Act"].astype(str))
df['SceneLE'] = le.fit_transform(df["Scene"].astype(str))
df['LineLE'] = le.fit_transform(df["Line"].astype(str))
print(df.head(15))
#Classifaction based on Act, Scene, and Play
labels = np.array(df['PlayerLE'])
​
feature_list = np.array(df[['ActLE','SceneLE','PlayLE']])
​
from sklearn.model_selection import train_test_split
​
train_features, test_features, train_labels, test_labels = train_test_split(feature_list, labels, test_size=0.25, random_state=42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
#Random Forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train_features, train_labels)
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import tree
​
forest = RandomForestClassifier(n_estimators=100)
forest.fit(train_features,train_labels)
prediction_labels = forest.predict(test_features)
print("Accuracy: ", metrics.accuracy_score(test_labels,prediction_labels))
#Decision Tree
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(train_features,train_labels)
prediction_labels = model.predict(test_features)
print("Accuracy: ", metrics.accuracy_score(test_labels,prediction_labels))
#Classification based on Act, Scene, Play, and Line
labels = df['PlayerLE']
features = df[['ActLE','SceneLE','LineLE', 'PlayLE']]
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.20)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

#Random Forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train_features, train_labels)
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import tree
​
forest = RandomForestClassifier(n_estimators=100)
forest.fit(train_features,train_labels)
prediction_labels = forest.predict(test_features)
print("Accuracy: ", metrics.accuracy_score(test_labels,prediction_labels))
#Decision Tree¶
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(train_features,train_labels)
prediction_labels = model.predict(test_features)
print("Accuracy: ", metrics.accuracy_score(test_labels,prediction_labels))