import inline as inline
import matplotlib
# import pandas_profiling as pdp
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import resample

from sklearn.linear_model import LogisticRegressionCV
import sklearn.metrics as metrics

from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

df = pd.read_csv('dataset/data.csv')
df.head()

# Loading test data Test Data
df.Class.value_counts()

df.info()

# plot the histogram of a variable from the dataset to see the skewness
normal_records = df.Class == 0
fraud_records = df.Class == 1

plt.figure(figsize=(20, 60))
for n, col in enumerate(df.drop('Class', axis=1).columns):
    plt.subplot(10, 3, n + 1)
    sns.histplot(df[col][df.Class == 1], bins=50)
    sns.histplot(df[col][df.Class == 0], bins=50)
    plt.title(col, fontsize=17)
#plt.show()

df[['Time', 'Amount', 'Class']].groupby('Class').describe()

df.boxplot('Amount')

df[df.Class == 0].plot.scatter('Amount', 'Time')
df[df.Class == 1].plot.scatter('Amount', 'Time')

# df[df.Amount > 10000].shape

# There are 7 record in dataset the Ammount is greater than 10,000.00.
# with scatterplot we can see all of these transactions are belongs to non-fraudelent as well
df = df.drop(df[df.Amount > 10000].index, axis=0)

df.boxplot('Time')

x = df.drop('Class', axis=1)
y = df.Class.values

corr_matrix = x.corr()
plt.figure(figsize=(30, 30))
sns.heatmap(corr_matrix, annot=True)
#plt.show()

# Handling Inbalance data.
counts = df.Class.value_counts()
print(counts)
print(f'legimate {(counts[0] / sum(counts)) * 100}% and Fraudent {(counts[1] / sum(counts)) * 100}%')

reg_model = LogisticRegression(max_iter=200, random_state=12, solver='liblinear')
reg_model.fit(x, y)

# coefficient matrix
coefficients = pd.concat([pd.DataFrame(x.columns), pd.DataFrame(np.transpose(reg_model.coef_))], axis=1)
coefficients.columns = ['Feature', 'Importance Coefficient']
coefficients.sort_values(by='Importance Coefficient', inplace=True)

# Plotting coefficient values
plt.figure(figsize=(20, 5))
sns.barplot(x='Feature', y='Importance Coefficient', data=coefficients)
plt.title("Logistic Regression with L2 Regularisation Feature Importance", fontsize=18)

#plt.show()

x.drop(['Time', 'Amount'], axis=1, inplace=True)

# Since dataset is highly unbalanced we can use under sampling or mix of under and over sampling to increase number
# of samples
leg_df = df[df.Class == 0]
fraud_df = df[df.Class == 1]

no_of_samples = round(leg_df.shape[0] * 0.05)
# no_of_samples


leg_df_2 = resample(leg_df, n_samples=no_of_samples, random_state=15)
# leg_df_2.describe()
df_sampled = pd.concat([leg_df_2, fraud_df], axis=0)

x_sampled = df_sampled.drop('Class', axis=1)
y_sampled = df_sampled.Class

ros = RandomOverSampler(random_state=42)

x, y = ros.fit_resample(x_sampled, y_sampled)

y.value_counts()

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=12)
y_train.value_counts(), y_test.value_counts()

columns = ['Model', 'accuracy score', ' Precision', 'Recall', 'f1_score']
evaluation_df = pd.DataFrame(columns=columns)

# evaluation_df

def print_results(model_name, y_test, y_pred, pred_prob=None):
    print(model_name)
    print('--------------------------------------------------------------------------')

    precision_score = metrics.precision_score(y_test, y_pred)
    recall_score = metrics.recall_score(y_test, y_pred)

    accuracy_score = metrics.accuracy_score(y_test, y_pred)
    print(f'accuracy score :{accuracy_score}')

    f1_score = metrics.f1_score(y_test, y_pred)

    classification_report = metrics.classification_report(y_test, y_pred)
    print(classification_report)

    #   save scores into dataframe for comparison
    evaluation_df.loc[len(evaluation_df.index)] = [model_name, accuracy_score, precision_score, recall_score, f1_score]

    Plot_confusion_matrix(y_test, y_pred, model_name)

    if pred_prob is not None:
        Plot_roc_curve(y_test, pred_prob, model_name, accuracy_score)


# Created a common function to plot confusion matrix
def Plot_confusion_matrix(y, pred, model_name):
    cm = metrics.confusion_matrix(y, pred)
    plt.clf()
    plt.imshow(cm, cmap=plt.cm.Accent)
    categoryNames = ['Non-Fraudulent', 'Fraudulent']
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    ticks = np.arange(len(categoryNames))
    plt.xticks(ticks, categoryNames, rotation=45)
    plt.yticks(ticks, categoryNames)
    s = [['TN', 'FP'], ['FN', 'TP']]

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(s[i][j]) + " = " + str(cm[i][j]), fontsize=12)
    #plt.show()


def Plot_roc_curve(y, y_prob, model_name, score):
    plt.title(f'ROC Curve - {model_name}')
    fpr, tpr, thresholds = metrics.roc_curve(y, y_prob)
    plt.plot(fpr, tpr, label="Test, auc=" + str(score))
    plt.legend(loc=4)
    #plt.show()


lr_model = LogisticRegression(max_iter=200, random_state=12)
lr_model.fit(x_train, y_train)
pred1 = lr_model.predict(x_test)
prob1 = lr_model.predict_proba(x_test)
print_results("Logistic Regression", y_test, pred1, prob1[:, -1])

cv_num = KFold(n_splits=10, shuffle=True, random_state=12)
lr_modelCV = LogisticRegressionCV(max_iter=200, penalty='l2', scoring='roc_auc', cv=cv_num, tol=10, random_state=12)
lr_modelCV.fit(x_train, y_train)
pred2 = lr_modelCV.predict(x_test)
prob2 = lr_modelCV.predict_proba(x_test)
print_results("Logistic Regression CV", y_test, pred2, prob2[:, -1])

gnb_model = BernoulliNB()
gnb_model.fit(x_train, y_train)
pred3 = gnb_model.predict(x_test)
prob3 = gnb_model.predict_proba(x_test)
print_results("Bernoulli Naive Bayes", y_test, pred3, prob3[:, -1])

rfc_model = RandomForestClassifier(bootstrap=True,
                                   max_features='sqrt', random_state=12)
rfc_model.fit(x_train, y_train)
pred5 = rfc_model.predict(x_test)
prob5 = rfc_model.predict_proba(x_test)
print_results("Random Forest Classifier + gini", y_test, pred5, prob5[:, -1])

rfc_model2 = RandomForestClassifier(bootstrap=True, criterion='entropy', max_features='sqrt', random_state=12)
rfc_model2.fit(x_train, y_train)
pred6 = rfc_model2.predict(x_test)
prob6 = rfc_model2.predict_proba(x_test)
print_results("Random Forest Classifier + entropy ", y_test, pred6, prob6[:, -1])

bcf_model = BaggingClassifier(DecisionTreeClassifier(),
                              n_estimators=200,
                              max_samples=0.8,
                              max_features=0.8,
                              oob_score=True,
                              random_state=12)
bcf_model.fit(x_train, y_train)
pred2 = bcf_model.predict(x_test)
print_results("Bagging Classifier", y_test, pred2)

gbc_model = GradientBoostingClassifier()
gbc_model.fit(x_train, y_train)
pred = gbc_model.predict(x_test)
pred_prob = gbc_model.predict_proba(x_test)
print_results("Gradient Boosting Classifier", y_test, pred)

adb_model = AdaBoostClassifier(n_estimators=200, random_state=12)
adb_model.fit(x_train, y_train)
pred = adb_model.predict(x_test)
print_results("Ada Boost Classifier", y_test, pred)

cv = KFold(n_splits=10, random_state=12, shuffle=True)
model = XGBClassifier(cv=cv, learning_rate=0.01)

gbc_model.fit(x_train, y_train)
pred = gbc_model.predict(x_test)
pred_prob = gbc_model.predict_proba(x_test)
print_results("Gradient Boosting Classifier", y_test, pred)


def create_model():
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=[30]))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


classifier = create_model()
classifier.summary()

early_stopping = EarlyStopping(patience=20, min_delta=0.001,
                               restore_best_weights=True)

scaller = StandardScaler()

x_train_scaled = scaller.fit_transform(x_train)
x_test_scaled = scaller.transform(x_test)

history = classifier.fit(x_train_scaled,
                         y_train,
                         epochs=500,
                         validation_split=0.25,
                         callbacks=[early_stopping],
                         verbose=1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
#plt.show()

y_pred = classifier.predict(x_test_scaled)

for i in range(len(y_pred)):
    if y_pred[i] > 0.5:
        y_pred[i] = 1
    else:
        y_pred[i] = 0

#Get Fraud Transactions
fraud_transactions = pd.DataFrame(fraud_df)

# saving the dataframe
fraud_transactions.to_csv('detected/fraud_set.csv')


#ANN - Artificial Neural Network
print_results("ANN ", y_test, y_pred)
