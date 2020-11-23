#imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#dataset
df = pd.read_csv("train.csv")
print(df.head())

#Check the amount of missing values per column
print(df.isnull().sum())

num_1 = len(df.loc[df['Survived'] == 1])
num_0 = len(df.loc[df['Survived'] == 0])
print("Number of survivors: {0} ({1:2.2f}%)".format(num_1, (num_1/ (num_1 + num_0)) * 100))
print("Number of deaths: {0} ({1:2.2f}%)".format(num_0, (num_0/ (num_1 + num_0)) * 100))

#Transform the strings of the 'Sex' column into numbers
df['Sex'] =df['Sex'].map({'female': 1, 'male': 0})
num_true = len(df.loc[df['Sex'] == True])
num_false = len(df.loc[df['Sex'] == False])
print("Number of women: {0} ({1:2.2f}%)".format(num_true, (num_true/ (num_true + num_false)) * 100))
print("Número of men   : {0} ({1:2.2f}%)".format(num_false, (num_false/ (num_true + num_false)) * 100))

#Transform the strings of the 'Embarked' column into numbers
df['Embarked'] =df['Embarked'].map({'S': 2,'C': 1, 'Q': 0})
num_S = len(df.loc[df['Embarked'] == 2])
num_C = len(df.loc[df['Embarked'] == 1])
num_Q = len(df.loc[df['Embarked'] == 0])
print('Number of embarked S: {0} ({1:2.2f}%)'.format(num_S, (num_S/ (num_S + num_C + num_Q)) * 100)) 
print('Number of embarked C: {0} ({1:2.2f}%)'.format(num_C, (num_C/ (num_S + num_C + num_Q)) * 100))
print('Number of embarked Q: {0} ({1:2.2f}%)'.format(num_Q, (num_Q/ (num_S + num_C + num_Q)) * 100))

#Dealing with empty data and checking if it worked
df['Age'].fillna(df['Age'].dropna().median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].dropna().median(), inplace=True)
print(df.isnull().sum())


#Graphs

#Correlation between some columns
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin']
train = df.drop(drop_elements, axis = 1)
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Correlation', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()

#Simple example of plotting collumns 
plt.hist(df['Age'], label="Age")
plt.grid(True)
plt.legend()
plt.show()

plt.hist(df['Pclass'], label = "Class")
plt.grid(True)
plt.legend()
plt.show()



#Splitting

#Splitting data between training and testing
import sklearn as sk
from sklearn.model_selection import train_test_split
#Feature Selection
features = ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
# Variable to be predicted
feature_prev = ['Survived']
# Objects
X = df[features].values
Y = df[feature_prev].values

split_test_size = 0.30
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = split_test_size, random_state = 42)
print("{0:0.2f}% train data".format((len(X_train)/len(df.index)) * 100))
print("{0:0.2f}% teste data".format((len(X_test)/len(df.index)) * 100))


#Building and training the model

#Using Naive Bayes
print("++++++++++++++++++++++++Naive Bayes++++++++++++++++++++++++")
from sklearn.naive_bayes import GaussianNB
modelo_v1 = GaussianNB()
modelo_v1.fit(X_train, Y_train.ravel())
from sklearn import metrics
nb_predict_train = modelo_v1.predict(X_train)
print("Accuracy Train: {0:.4f}".format(metrics.accuracy_score(Y_train, nb_predict_train)))
nb_predict_test = modelo_v1.predict(X_test)
print("Accuracy Test: {0:.4f}".format(metrics.accuracy_score(Y_test, nb_predict_test)))
print("Classification Report")
print(metrics.classification_report(Y_test, nb_predict_test, labels = [1, 0]))

#Using Random Forest
print("++++++++++++++++++++++++RANDOM FOREST++++++++++++++++++++++++")
from sklearn.ensemble import RandomForestClassifier
modelo_v2 = RandomForestClassifier(random_state = 42)
modelo_v2.fit(X_train, Y_train.ravel())
rf_predict_train = modelo_v2.predict(X_train)
print("Accuracy Train: {0:.4f}".format(metrics.accuracy_score(Y_train, rf_predict_train)))
rf_predict_test = modelo_v2.predict(X_test)
print("Accuracy Test: {0:.4f}".format(metrics.accuracy_score(Y_test, rf_predict_test)))
print()
print("Classification Report")
print(metrics.classification_report(Y_test, rf_predict_test, labels = [1, 0]))

#Using Logistic Regression
print("++++++++++++++++++++++++LOGISTIC REGRESSION++++++++++++++++++++++++")
from sklearn.linear_model import LogisticRegression
modelo_v3 = LogisticRegression(C = 0.7, random_state = 42, max_iter = 1000)
modelo_v3.fit(X_train, Y_train.ravel())
lr_predict_train = modelo_v1.predict(X_train)
print("Accuracy Train: {0:.4f}".format(metrics.accuracy_score(Y_train, lr_predict_train)))
lr_predict_test = modelo_v3.predict(X_test)
print("Accuracy Test: {0:.4f}".format(metrics.accuracy_score(Y_test, lr_predict_test)))
print()
print("Classification Report")
print(metrics.classification_report(Y_test, lr_predict_test, labels = [1, 0]))

print("++++++++++++++++++++++++End of Train++++++++++++++++++++++++")

print("++++++++++++++++++++++++Begin of Test++++++++++++++++++++++++")
#Read the file, organize the data, apply the selected method
df2 = pd.read_csv("test.csv")
print(df2.head())

df2['Sex'] =df2['Sex'].map({'female': 1, 'male': 0})
df2['Embarked'] =df2['Embarked'].map({'S': 2,'C': 1, 'Q': 0})
df2['Age'].fillna(df2['Age'].dropna().median(), inplace=True)
df2['Embarked'].fillna(df2['Embarked'].dropna().median(), inplace=True)
df2['Fare'].fillna(df2['Fare'].dropna().median(), inplace=True)
X2_test=pd.get_dummies(df2[features])

modelo_v4 = RandomForestClassifier(random_state = 42)
modelo_v4.fit(X_train, Y_train.ravel())
predict_test = modelo_v4.predict(X2_test)

output = pd.DataFrame({'PassengerId': df2.PassengerId, 'Survived': predict_test})
output.to_csv('my_submission.csv', index=False)
print("Your submission was saved!")
