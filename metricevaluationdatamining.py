#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip3 install pandas
#pip3 install seaborn
#pip3 install sklearn
#pip3 install matplotlib


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score


# # Reading the Data

# In[3]:


diabetes=pd.read_csv('Desktop/Diabetes-Prediction-master/Diabetes-Prediction-master/diabetes.csv') # Please 
                                                       # update the path according to your system
                                                                            

diabetes.head()


#  # Data Cleaning process and preparation Stage

# In[4]:


# Let's have a look at the Dimensions of the data set
print(diabetes.shape)


# In[5]:


# We will now remove unusual rows of data
diabetes_mod = diabetes[(diabetes.BloodPressure != 0) & (diabetes.BMI != 0) & (diabetes.Glucose != 0)]

# Dimensions of data set after cleansing are as follows using the below command
print(diabetes_mod.shape)


# # Feature Selection Stage

# In[6]:


# Features/Response
feature_names = ['Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction']
X = diabetes_mod[feature_names]
y = diabetes_mod.Outcome


# # Histogram for the Diabetes Data

# In[7]:


diabetes.hist(bins=10,figsize=(10,10))
plt.show()


# # Heat Map Generation

# In[8]:


#correlation

sns.heatmap(diabetes.corr())
# we can see skin thickness,insulin,pregnencies and age are full independent to each other
#age and pregencies has negative correlation


# In[9]:


#lets count total outcome in each target 0 1
#0 means no diabetes
#1 means patient with diabtes
sns.countplot(y=diabetes['Outcome'],palette='Set1')


# In[10]:


sns.set(style="ticks")
sns.pairplot(diabetes, hue="Outcome")


# # Outlier Box Plot Visualization

# In[11]:


sns.set(style="whitegrid")
diabetes.boxplot(figsize=(15,6))


# In[12]:


#box plot
sns.set(style="whitegrid")

sns.set(rc={'figure.figsize':(4,2)})
sns.boxplot(x=diabetes['Insulin'])
plt.show()
sns.boxplot(x=diabetes['BloodPressure'])
plt.show()
sns.boxplot(x=diabetes['DiabetesPedigreeFunction'])
plt.show()


# # Outlier Removal

# In[13]:


Q1=diabetes.quantile(0.25)
Q3=diabetes.quantile(0.75)
IQR=Q3-Q1

print("---Q1--- \n",Q1)
print("\n---Q3--- \n",Q3)
print("\n---IQR---\n",IQR)

print((diabetes < (Q1 - 1.5 * IQR))|(diabetes > (Q3 + 1.5 * IQR)))


# In[14]:


diabetes_out = diabetes[~((diabetes < (Q1 - 1.5 * IQR)) |(diabetes > (Q3 + 1.5 * IQR))).any(axis=1)]
diabetes.shape,diabetes_out.shape
#We see that many records are deleted after outlier removal


# # Scatter Matrix after outlier removal

# In[15]:


sns.set(style="ticks")
sns.pairplot(diabetes_out, hue="Outcome")
plt.show()


# # Random Forest Classifier

# In[16]:


from sklearn.ensemble import RandomForestClassifier


# In[17]:


randomforest = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0)


# # Train / Test Split

# In[18]:


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 0)

randomforest.fit(X_train, y_train)
y_pred = randomforest.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy {}".format(accuracy))


# # K-Fold Cross Validation

# In[19]:


accuracy = cross_val_score(randomforest, X, y, cv = 10, scoring='accuracy')
accuracy.mean()
print("Accuracy {}".format(accuracy))


# In[20]:


print("Accuracy: %0.2f (+/- %0.2f)" % (accuracy.mean(), accuracy.std() * 2))


# # Confusion Matrix

# In[21]:


from sklearn.metrics import confusion_matrix


# In[22]:


# Method to plot the confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.rcParams["figure.figsize"] = (35,30)
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# In[23]:


confusion = confusion_matrix(y_test, y_pred)
print(confusion)


# In[24]:


plot_confusion_matrix(confusion, classes=['Non Diabetic', 'Diabetic'], title='Confusion matrix')


# # Values of TP, TN, FP, FN based on confusion matrix

# In[25]:


# True Positives
TP = confusion[1, 1]
print("The value of TP is :" + str(TP))

# True Negatives
TN = confusion[0, 0] 
print("The value of TN is :" + str(TN))
# False Positives
FP = confusion[0, 1] 
print("The value of FP is :" + str(FP))
# False Negatives
FN = confusion[1, 0]
print("The value of FN is :" + str(FN))


# # Number of Positive examples

# In[26]:


P = (TP + FN)
print(P)


# # Number of Negative Examples

# In[27]:


N = (TN + FP)
print(N)


# # Evaluating various metrics based on Confusion Matrix

# In[28]:


from sklearn.metrics import recall_score, precision_score


# # Classification Accuracy

# In[29]:


print((TP + TN) / float(TP + TN + FP + FN))
print(accuracy_score(y_test, y_pred))
accuracy = cross_val_score(randomforest, X, y, cv = 10, scoring='accuracy')
accuracy.mean()
print("Accuracy {}".format(accuracy))


# In[30]:


print("Accuracy: %0.2f (+/- %0.2f)" % (accuracy.mean(), accuracy.std() * 2))


# # Sensitivity or TPR (True Positive Rate)

# In[31]:


print(TP / float(TP + FN))
print(recall_score(y_test, y_pred))
recall = cross_val_score(randomforest, X, y, cv = 10, scoring='recall')
recall.mean()
print("Recall {}".format(recall))


# In[32]:


print("Recall: %0.2f (+/- %0.2f)" % (recall.mean(), recall.std() * 2))


# # Specificity or TNR (True Negative Rate)

# In[33]:


print(TN / float(TN + FP))


# # False Positive Rate

# In[34]:


FPR = (FP / (TN + FP))
print(FPR)


# # False Negative Rate

# In[35]:


FNR = ((FN) / (TP + FN))
print(FNR)


# # Precision

# In[36]:


print(TP / float(TP + FP))
print(precision_score(y_test, y_pred))
precision = cross_val_score(randomforest, X, y, cv = 10, scoring='precision')
precision.mean()
print("Precision {}".format(precision))


# In[37]:


print("Precision: %0.2f (+/- %0.2f)" % (precision.mean(), precision.std() * 2))


# # Heidke Skill Score

# In[38]:


HSS = 2 * (TP * TN - FP * FN) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))
print (HSS)


# # Balanced Accuracy (BACC)

# In[39]:


BACC = 1/2 * ((TP / (TP + FN)) + ((TN) / (FP + TN)))
print(BACC)
balanced_accuracy = cross_val_score(randomforest, X, y, cv = 10, scoring='balanced_accuracy')
balanced_accuracy.mean()
print("Balanced_accuracy {}".format(balanced_accuracy))


# In[40]:


print("Balanced_Accuracy: %0.2f (+/- %0.2f)" % (balanced_accuracy.mean(), balanced_accuracy.std() * 2))


# # True Skill Statistics

# In[41]:


TSS = (((TP) / (TP + FN)) - ((FP) / (FP + TN)) )
print(TSS)


# # F1 measure (F1)

# In[42]:


F1 = ((2 * TP)/ (2 * (TP + FP + FN)))
print(F1)
f1 = cross_val_score(randomforest, X, y, cv = 10, scoring='f1')
balanced_accuracy.mean()
print("F1_score {}".format(f1))


# In[43]:


print("F1_score: %0.2f (+/- %0.2f)" % (f1.mean(), f1.std() * 2))


# # Error Rate

# In[44]:


errorrate = ((FP + FN) / (TP + FP + TN + FN))
print(errorrate)


# # Negative Predicted Value

# In[45]:


NPV = (TN / (TN + FN))
print(NPV)


# # False Discovery Rate

# In[46]:


FDR = (FP / (FP + TP))
print(FDR)


# # Handling the Classification Threshold

# In[47]:


# print the first 10 predicted responses
randomforest.predict(X_test)[0:10]


# In[48]:


# print the first 10 predicted probabilities of class membership
randomforest.predict_proba(X_test)[0:10, :]


# In[49]:


# store the predicted probabilities for class 1 (diabetic)
y_pred_prob = randomforest.predict_proba(X_test)[:, 1]


# In[50]:


# histogram of predicted probabilities
plt.hist(y_pred_prob, bins=8, linewidth=1.2)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of diabetes')
plt.ylabel('Frequency')


# In[51]:


# predict diabetes if the predicted probability is greater than 0.3
from sklearn.preprocessing import binarize

y_pred_class = binarize([y_pred_prob], 0.3)[0]


# In[52]:


# previous confusion matrix (default threshold of 0.5)
print(confusion)


# In[53]:


# new confusion matrix (threshold of 0.3)
confusion_new = confusion_matrix(y_test, y_pred_class)
print(confusion_new)


# # Values of TP, TN, FP, FN based on new confusion matrix

# In[54]:


# True Positives
TP = confusion_new[1, 1]
print("The value of TP is :" + str(TP))
# True Negatives
TN = confusion_new[0, 0] 
print("The value of TN is :" + str(TN))
# False Positives
FP = confusion_new[0, 1] 
print("The value of FP is :" + str(FP))
# False Negatives
FN = confusion_new[1, 0]
print("The value of FN is :" + str(FN))


# In[55]:


# We observe that the sensitivity has increased
print(TP / float(TP + FN))
print(recall_score(y_test, y_pred_class))


# In[56]:


# specificity has decreased 
print(TN / float(TN + FP))


# # ROC and AUC (Area Under the Curves)

# In[57]:


from sklearn.metrics import roc_curve, roc_auc_score


# In[58]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[59]:


roc_auc = cross_val_score(randomforest, X, y, cv = 10, scoring='roc_auc')
roc_auc.mean()
print("roc_auc {}".format(roc_auc))


# In[60]:


print("ROC_AUC: %0.2f (+/- %0.2f)" % (roc_auc.mean(), roc_auc.std() * 2))


# In[61]:


# define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])


# In[62]:


evaluate_threshold(0.3)


# In[63]:


evaluate_threshold(0.5)


# In[64]:


train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2)


# In[65]:


train_X.shape,test_X.shape,train_y.shape,test_y.shape


# In[66]:


from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate

def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]

#cross validation purpose
scoring = {'accuracy': make_scorer(accuracy_score),'prec': 'precision'}
scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
           'fp': make_scorer(fp), 'fn': make_scorer(fn)}

def display_result(result):
    print("TP: ",result['test_tp'])
    print("TN: ",result['test_tn'])
    print("FN: ",result['test_fn'])
    print("FP: ",result['test_fp'])


# In[67]:


acc=[]
roc=[]

clf=RandomForestClassifier()
clf.fit(train_X,train_y)
y_pred=clf.predict(test_X)
#find accuracy
ac=accuracy_score(test_y,y_pred)
acc.append(ac)

#find the ROC_AOC curve
rc=roc_auc_score(test_y,y_pred)
roc.append(rc)
print("\nAccuracy {0} ROC {1}".format(ac,rc))

#cross val score
result=cross_validate(clf,train_X,train_y,scoring=scoring,cv=10)
display_result(result)
pd.DataFrame(data={'Actual':test_y,'Predicted':y_pred}).head()


# In[68]:


tp = np.append(result["test_tp"], sum(result["test_tp"])/10)
tn = np.append(result["test_tn"], sum(result["test_tn"])/10)
fp = np.append(result["test_fp"], sum(result["test_fp"])/10)
fn = np.append(result["test_fn"], sum(result["test_fn"])/10)
acc = np.append(accuracy,accuracy.mean())
recc = np.append(recall , recall.mean())
prec = np.append(precision,precision.mean())
bal_acc = np.append(balanced_accuracy,balanced_accuracy.mean())
f1_score = np.append(f1,f1.mean())
roc = np.append(roc_auc,roc_auc.mean())


# In[69]:


data = { 'TP':tp,
        'TN':tn,
        'FP':fp,
        'FN':fn,
        'Accuracy':acc,
        'Recall/Sensitivity':recc,
        'Precision':prec,
        'Balanced Accuracy':bal_acc,
        'f1_score':f1_score,
        'ROC_AUC':roc
       }
df = pd.DataFrame(data,index = ['fold 1','fold 2','fold 3','fold 4','fold 5','fold 6','fold 7','fold 8','fold 9','fold 10','avg'])
df
# result


# # Naive Bayes Classifier

# In[70]:


from sklearn.naive_bayes import GaussianNB


# In[71]:


naivebayes = GaussianNB( priors=None, var_smoothing=1e-09)


# # Train / Test Split

# In[72]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 0)

naivebayes.fit(X_train, y_train)
y_pred = naivebayes.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy {}".format(accuracy))


# # K-Fold Cross Validation

# In[73]:


accuracy = cross_val_score(naivebayes, X, y, cv = 10, scoring='accuracy')
accuracy.mean()
print("Accuracy {}".format(accuracy))


# In[74]:


print("Accuracy: %0.2f (+/- %0.2f)" % (accuracy.mean(), accuracy.std() * 2))


# # Confusion Matrix

# In[75]:


from sklearn.metrics import confusion_matrix


# In[76]:


# Method to plot the confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# In[77]:


confusion = confusion_matrix(y_test, y_pred)
print(confusion)


# In[78]:


plot_confusion_matrix(confusion, classes=['Non Diabetic', 'Diabetic'], title='Confusion matrix')


# # Values of TP, TN, FP, FN based on confusion matrix

# In[79]:


# True Positives
TP = confusion[1, 1]
print("The value of TP is :" + str(TP))
# True Negatives
TN = confusion[0, 0] 
print("The value of TN is :" + str(TN))
# False Positives
FP = confusion[0, 1] 
print("The value of FP is :" + str(FP))
# False Negatives
FN = confusion[1, 0]
print("The value of FN is :" + str(FN))


# # Number of positive Examples

# In[80]:


P = (TP + FN)
print(P)


# # Number of negative examples

# In[81]:


N = (TN + FP)
print(N)


# # Evaluating various metrics based on confusion matrix

# In[82]:


from sklearn.metrics import recall_score, precision_score


# # Classification Accuracy

# In[83]:


print((TP + TN) / float(TP + TN + FP + FN))
print(accuracy_score(y_test, y_pred))
accuracy = cross_val_score(naivebayes, X, y, cv = 10, scoring='accuracy')
accuracy.mean()
print("Accuracy {}".format(accuracy))


# In[84]:


print("Accuracy: %0.2f (+/- %0.2f)" % (accuracy.mean(), accuracy.std() * 2))


# # Sensitivity or TPR (True Positive Rate)

# In[85]:


print(TP / float(TP + FN))
print(recall_score(y_test, y_pred))
recall = cross_val_score(naivebayes, X, y, cv = 10, scoring='recall')
recall.mean()
print("Recall {}".format(recall))


# In[86]:


print("Recall: %0.2f (+/- %0.2f)" % (recall.mean(), recall.std() * 2))


# # Specificity or TNR (True Negative Rate)

# In[87]:


print(TN / float(TN + FP))


# # False Positive Rate

# In[88]:


print(FP / float(TN + FP))


# # False Negative Rate

# In[89]:


FNR = ((FN) / (TP + FN))
print(FNR)


# # Precision

# In[90]:


print(TP / float(TP + FP))
print(precision_score(y_test, y_pred))
precision = cross_val_score(naivebayes, X, y, cv = 10, scoring='precision')
precision.mean()
print("Precision {}".format(precision))


# In[91]:


print("Precision: %0.2f (+/- %0.2f)" % (precision.mean(), precision.std() * 2))


# # Heidke Skill Score

# In[92]:


HSS = 2 * (TP * TN - FP * FN) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))
print (HSS)


# # Balanced Accuracy (BACC)

# In[93]:


BACC = 1/2 * ((TP / (TP + FN)) + ((TN) / (FP + TN)))
print(BACC)
balanced_accuracy = cross_val_score(naivebayes, X, y, cv = 10, scoring='balanced_accuracy')
balanced_accuracy.mean()
print("Balanced_Accuracy {}".format(balanced_accuracy))


# In[94]:


print("Balanced_Accuracy: %0.2f (+/- %0.2f)" % (balanced_accuracy.mean(), balanced_accuracy.std() * 2))


#  # True Skill Statistics (TSS)

# In[95]:


TSS = (((TP) / (TP + FN)) - ((FP) / (FP + TN)) )
print(TSS)


# # F1 measure (F1)

# In[96]:


F1 = ((2 * TP)/ (2 * (TP + FP + FN)))
print(F1)
f1 = cross_val_score(naivebayes, X, y, cv = 10, scoring='f1')
f1.mean()
print("F1_Score {}".format(f1))


# In[97]:


print("F1_SCORE: %0.2f (+/- %0.2f)" % (f1.mean(), f1.std() * 2))


# # Error Rate

# In[98]:


errorrate = ((FP + FN) / (TP + FP + TN + FN))
print(errorrate)


# # Negative Predictive Value

# In[99]:


NPV = (TN / (TN + FN))
print(NPV)


# # False Discovery Rate

# In[100]:


FDR = (FP / (FP + TP))
print(FDR)


# # Handling the Classification Threshold

# In[101]:


# Printing the first 10 predicted responses
naivebayes.predict(X_test)[0:10]


# In[102]:


# print the first 10 predicted probabilities of class membership
naivebayes.predict_proba(X_test)[0:10, :]


# In[103]:


# store the predicted probabilities for class 1 (diabetic)
y_pred_prob = naivebayes.predict_proba(X_test)[:, 1]


# In[104]:


# histogram of predicted probabilities
plt.hist(y_pred_prob, bins=8, linewidth=1.2)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of diabetes')
plt.ylabel('Frequency')


# In[105]:


# predict diabetes if the predicted probability is greater than 0.3
from sklearn.preprocessing import binarize

y_pred_class = binarize([y_pred_prob], 0.3)[0]


# In[106]:


# previous confusion matrix (default threshold of 0.5)
print(confusion)


# In[107]:


# new confusion matrix (threshold of 0.3)
confusion_new = confusion_matrix(y_test, y_pred_class)
print(confusion_new)


# # Values of TP, TN, FP, FN based on new confusion matrix

# In[108]:


# True Positives
TP = confusion_new[1, 1]
print("The value of TP is :" + str(TP))
# True Negatives
TN = confusion_new[0, 0] 
print("The value of TN is :" + str(TN))
# False Positives
FP = confusion_new[0, 1] 
print("The value of FP is :" + str(FP))
# False Negatives
FN = confusion_new[1, 0]
print("The value of FN is :" + str(FN))


# In[109]:


# sensitivity has increased
print(TP / float(TP + FN))
print(recall_score(y_test, y_pred_class))


# In[110]:


# specificity has decreased
print(TN / float(TN + FP))


# # ROC and AUC (Area Under the Curve)

# In[111]:


from sklearn.metrics import roc_curve, roc_auc_score


# In[112]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[113]:


roc_auc = cross_val_score(naivebayes, X, y, cv = 10, scoring='roc_auc')
roc_auc.mean()
print("roc_auc {}".format(roc_auc))


# In[114]:


print("ROC_AUC: %0.2f (+/- %0.2f)" % (roc_auc.mean(), roc_auc.std() * 2))


# In[115]:


# define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])


# In[116]:


evaluate_threshold(0.3)


# In[117]:


evaluate_threshold(0.5)


# In[118]:


train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2)


# In[119]:


train_X.shape,test_X.shape,train_y.shape,test_y.shape


# In[120]:


from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate

def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]

#cross validation purpose
scoring = {'accuracy': make_scorer(accuracy_score),'prec': 'precision'}
scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
           'fp': make_scorer(fp), 'fn': make_scorer(fn)}

def display_result(result):
    print("TP: ",result['test_tp'])
    print("TN: ",result['test_tn'])
    print("FN: ",result['test_fn'])
    print("FP: ",result['test_fp'])


# In[121]:


acc=[]
roc=[]

clf=GaussianNB()
clf.fit(train_X,train_y)
y_pred=clf.predict(test_X)
#find accuracy
ac=accuracy_score(test_y,y_pred)
acc.append(ac)

#find the ROC_AOC curve
rc=roc_auc_score(test_y,y_pred)
roc.append(rc)
print("\nAccuracy {0} ROC {1}".format(ac,rc))

#cross val score
result=cross_validate(clf,train_X,train_y,scoring=scoring,cv=10)
display_result(result)
pd.DataFrame(data={'Actual':test_y,'Predicted':y_pred}).head()


# In[122]:


tp = np.append(result["test_tp"], sum(result["test_tp"])/10)
tn = np.append(result["test_tn"], sum(result["test_tn"])/10)
fp = np.append(result["test_fp"], sum(result["test_fp"])/10)
fn = np.append(result["test_fn"], sum(result["test_fn"])/10)
acc = np.append(accuracy,accuracy.mean())
recc = np.append(recall , recall.mean())
prec = np.append(precision,precision.mean())
bal_acc = np.append(balanced_accuracy,balanced_accuracy.mean())
f1_score = np.append(f1,f1.mean())
roc = np.append(roc_auc,roc_auc.mean())


# In[123]:


data = { 'TP':tp,
        'TN':tn,
        'FP':fp,
        'FN':fn,
        'Accuracy':acc,
        'Recall/Sensitivity':recc,
        'Precision':prec,
        'Balanced Accuracy':bal_acc,
        'f1_score':f1_score,
        'ROC_AUC':roc
       }
df = pd.DataFrame(data,index = ['fold 1','fold 2','fold 3','fold 4','fold 5','fold 6','fold 7','fold 8','fold 9','fold 10','avg'])
df
# result


# # K- Nearest Neighbour Classification

# In[124]:


from sklearn.neighbors import KNeighborsClassifier


# In[125]:


knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)


# # Train / Test Split

# In[126]:


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 0)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy {}".format(accuracy))


# # K-Fold Cross Validation

# In[127]:


accuracy = cross_val_score(knn, X, y, cv = 10, scoring='accuracy')
accuracy.mean()
print("Accuracy {}".format(accuracy))


# In[128]:


print("Accuracy: %0.2f (+/- %0.2f)" % (accuracy.mean(), accuracy.std() * 2))


# # Confusion Matrix

# In[129]:


from sklearn.metrics import confusion_matrix


# In[130]:


# Method to plot the confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# In[131]:


confusion = confusion_matrix(y_test, y_pred)
print(confusion)


# In[132]:


plot_confusion_matrix(confusion, classes=['Non Diabetic', 'Diabetic'], title='Confusion matrix')


# # Values of TP, TN, FP, FN based on confusion matrix

# In[133]:


# True Positives
TP = confusion[1, 1]
print("The value of TP is :" + str(TP))
# True Negatives
TN = confusion[0, 0] 
print("The value of TN is :" + str(TN))
# False Positives
FP = confusion[0, 1] 
print("The value of FP is :" + str(FP))
# False Negatives
FN = confusion[1, 0]
print("The value of FN is :" + str(FN))


# # Number of positive examples

# In[134]:


P = (TP + FN)
print(P)


# # Number of negative examples

# In[135]:


N = (TN + FP)
print(N)


# # Evaluating various metrics based on confusion matrix

# In[136]:


from sklearn.metrics import recall_score, precision_score


# # Classification Accuracy

# In[137]:


print((TP + TN) / float(TP + TN + FP + FN))
print(accuracy_score(y_test, y_pred))
accuracy = cross_val_score(knn, X, y, cv = 10, scoring='accuracy')
accuracy.mean()
print("Accuracy {}".format(accuracy))


# In[138]:


print("Accuracy: %0.2f (+/- %0.2f)" % (accuracy.mean(), accuracy.std() * 2))


# # Sensitivity or TPR (True Positive Rate)

# In[139]:


print(TP / float(TP + FN))
print(recall_score(y_test, y_pred))
recall = cross_val_score(knn, X, y, cv = 10, scoring='recall')
recall.mean()
print("Recall {}".format(recall))


# In[140]:


print("Recall: %0.2f (+/- %0.2f)" % (recall.mean(), recall.std() * 2))


# # Specificity or TNR (True Negative Rate)

# In[141]:


print(TN / float(TN + FP))


# # False Positive Rate

# In[142]:


print(FP / float(TN + FP))


# # False Negative Rate

# In[143]:


FNR = ((FN) / (TP + FN))
print(FNR)


# # Precision

# In[144]:


print(TP / float(TP + FP))
print(precision_score(y_test, y_pred))
precision = cross_val_score(knn, X, y, cv = 10, scoring='precision')
precision.mean()
print("Accuracy {}".format(precision))


# In[145]:


print("Precision: %0.2f (+/- %0.2f)" % (precision.mean(), precision.std() * 2))


# # Heidke Skill Score

# In[146]:


HSS = 2 * (TP * TN - FP * FN) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))
print (HSS)


# # Balanced Accuracy (BACC)

# In[147]:


BACC = 1/2 * ((TP / (TP + FN)) + ((TN) / (FP + TN)))
print(BACC)
balanced_accuracy = cross_val_score(knn, X, y, cv = 10, scoring='balanced_accuracy')
balanced_accuracy.mean()
print("Balanced_Accuracy {}".format(balanced_accuracy))


# In[148]:


print("Balanced_Accuracy: %0.2f (+/- %0.2f)" % (balanced_accuracy.mean(), balanced_accuracy.std() * 2))


# # True Skill Statistics

# In[149]:


TSS = (((TP) / (TP + FN)) - ((FP) / (FP + TN)) )
print(TSS)


# # F1 measure (F1)

# In[150]:


F1 = ((2 * TP)/ (2 * (TP + FP + FN)))
print(F1)
f1 = cross_val_score(knn, X, y, cv = 10, scoring='f1')
f1.mean()
print("F1_Score {}".format(f1))


# In[151]:


print("F1_SCORE: %0.2f (+/- %0.2f)" % (f1.mean(), f1.std() * 2))


# # Error Rate

# In[152]:


errorrate = ((FP + FN) / (TP + FP + TN + FN))
print(errorrate)


# # Negative Predictive Value

# In[153]:


NPV = (TN / (TN + FN))
print(NPV)


# # False Discovery Rate

# In[154]:


FDR = (FP / (FP + TP))
print(FDR)


# # Handling the Classification Threshold

# In[155]:


# print the first 10 predicted responses
knn.predict(X_test)[0:10]


# In[156]:


# print the first 10 predicted probabilities of class membership
knn.predict_proba(X_test)[0:10, :]


# In[157]:


# store the predicted probabilities for class 1 (diabetic)
y_pred_prob = knn.predict_proba(X_test)[:, 1]


# In[158]:


# histogram of predicted probabilities
plt.hist(y_pred_prob, bins=8, linewidth=1.2)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of diabetes')
plt.ylabel('Frequency')


# In[159]:


# predict diabetes if the predicted probability is greater than 0.3
from sklearn.preprocessing import binarize

y_pred_class = binarize([y_pred_prob], 0.3)[0]


# In[160]:


# previous confusion matrix (default threshold of 0.5)
print(confusion)


# In[161]:


# new confusion matrix (threshold of 0.3)
confusion_new = confusion_matrix(y_test, y_pred_class)
print(confusion_new)


# # Values of TP, TN, FP, FN based on new confusion matrix

# In[162]:


# True Positives
TP = confusion_new[1, 1]
print("The value of TP is :" + str(TP))
# True Negatives
TN = confusion_new[0, 0] 
print("The value of TN is :" + str(TN))
# False Positives
FP = confusion_new[0, 1] 
print("The value of FP is :" + str(FP))
# False Negatives
FN = confusion_new[1, 0]
print("The value of FN is :" + str(FN))


# In[163]:


# sensitivity has increased
print(TP / float(TP + FN))
print(recall_score(y_test, y_pred_class))


# In[164]:


# specificity has decreased
print(TN / float(TN + FP))


# # ROC and AUC (Area Under the Curves)

# In[165]:


from sklearn.metrics import roc_curve, roc_auc_score


# In[166]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')


# In[167]:


roc_auc = cross_val_score(knn, X, y, cv = 10, scoring='roc_auc')
roc_auc.mean()
print("roc_auc {}".format(roc_auc))


# In[168]:


print("ROC_AUC: %0.2f (+/- %0.2f)" % (roc_auc.mean(), roc_auc.std() * 2))


# In[169]:


# define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])


# In[170]:


evaluate_threshold(0.3)


# In[171]:


evaluate_threshold(0.5)


# In[172]:


train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2)


# In[173]:


train_X.shape,test_X.shape,train_y.shape,test_y.shape


# In[174]:


from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate

def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]

#cross validation purpose
scoring = {'accuracy': make_scorer(accuracy_score),'prec': 'precision'}
scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
           'fp': make_scorer(fp), 'fn': make_scorer(fn)}

def display_result(result):
    print("TP: ",result['test_tp'])
    print("TN: ",result['test_tn'])
    print("FN: ",result['test_fn'])
    print("FP: ",result['test_fp'])


# In[175]:


acc=[]
roc=[]

clf=KNeighborsClassifier()
clf.fit(train_X,train_y)
y_pred=clf.predict(test_X)
#find accuracy
ac=accuracy_score(test_y,y_pred)
acc.append(ac)

#find the ROC_AOC curve
rc=roc_auc_score(test_y,y_pred)
roc.append(rc)
print("\nAccuracy {0} ROC {1}".format(ac,rc))

#cross val score
result=cross_validate(clf,train_X,train_y,scoring=scoring,cv=10)
display_result(result)
pd.DataFrame(data={'Actual':test_y,'Predicted':y_pred}).head()


# In[176]:


tp = np.append(result["test_tp"], sum(result["test_tp"])/10)
tn = np.append(result["test_tn"], sum(result["test_tn"])/10)
fp = np.append(result["test_fp"], sum(result["test_fp"])/10)
fn = np.append(result["test_fn"], sum(result["test_fn"])/10)
acc = np.append(accuracy,accuracy.mean())
recc = np.append(recall , recall.mean())
prec = np.append(precision,precision.mean())
bal_acc = np.append(balanced_accuracy,balanced_accuracy.mean())
f1_score = np.append(f1,f1.mean())
roc = np.append(roc_auc,roc_auc.mean())


# In[177]:


data = { 'TP':tp,
        'TN':tn,
        'FP':fp,
        'FN':fn,
        'Accuracy':acc,
        'Recall/Sensitivity':recc,
        'Precision':prec,
        'Balanced Accuracy':bal_acc,
        'f1_score':f1_score,
        'ROC_AUC':roc
       }
df = pd.DataFrame(data,index = ['fold 1','fold 2','fold 3','fold 4','fold 5','fold 6','fold 7','fold 8','fold 9','fold 10','avg'])
df
# result


# In[ ]:





# In[ ]:




