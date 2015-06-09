# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# #Classifying pediatric IBD stool samples (work in progress)

# <markdowncell>

# This notebook is a recoding of the analysis used in the PLoSONE paper: [Non-Invasive Mapping of the Gastrointestinal Microbiota Identifies Children with Inflammatory Bowel Disease](http://www.plosone.org/article/info%3Adoi%2F10.1371%2Fjournal.pone.0039242) using python, sklearn and pandas.
# 
# [We](http://almlab.mit.edu) decided that the SLiME package, as it was packaged for the publication of the paper, should not be available anymore. This notebook replaces it, replicating the analysis executed on the paper with more up-to-date tools and (hopefully soon) expanding on its conclusion. 
# I hope this can be the starting point for others trying to follow the same approach and improve upon it. 

# <codecell>

%matplotlib inline

# <codecell>

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import LabelEncoder, label_binarize

# <markdowncell>

# ##Loading data
# The data came from two rounds of 16S sequencing of previously collected stool samples. Here we will use the OTU tables directly, which were created by using the RDP classifier and were subsequently normalized (details in the paper's methods).
# 
# Sequencing was performed at the [Broad Institute](https://www.broadinstitute.org/). The first round of sequencing was dubbed CHIMP (Children Hospital IBD Pediatric), while the second round of sequencing -- performed following the request of an anonymous peer reviewer -- was termed 'blind validation'. Its purpose was to further validate the algorithm trained on the CHIMP dataset, as the reviewer did not think sufficient a "leave 20% out" approach on CHIMP was sufficient to demonstrate robust prediction. These were used as training and test set in the last figure of the paper respectively.
# 
# It is useful to join the two data sets here.

# <codecell>

#get the CHIMP training data

X_chimp = pd.read_csv('data/chimp/chimp.Qsorted.rdpout.xtab.norm', delimiter="\t", index_col=0)
y_chimp = pd.read_csv('data/chimp/sampledata.training.chimp.csv', index_col=0)

#just make sure the labels are the same
X_chimp.sort_index(inplace=True)
y_chimp.sort_index(inplace=True)
assert (X_chimp.index == y_chimp.index).all()

# <codecell>

## do the same for the blind validation test data
X_blind = pd.read_csv('data/chimp/blind.sorted.rdpout.xtab.norm',
                        delimiter="\t", index_col=0)
y_blind = pd.read_csv('data/chimp/sampledata.validation.blind.csv',
                        index_col=0)

X_blind.sort_index(inplace=True)
y_blind.sort_index(inplace=True)
assert (X_blind.index == y_blind.index).all()

# <codecell>

#concatenate using pandas
X = pd.concat([X_chimp, X_blind], keys=['chimp','blind'])
X.head()

# <codecell>

X.fillna(value=0,inplace=True) #replace NAs with zeroes

# <codecell>

y_dx = pd.concat([y_chimp.dx, y_blind.dx], keys=['chimp','blind'])
y_dx #btw, what joy is to use pandas over R/dplyr for this. so intuitive and fast.

# <codecell>

#convert the training and testing labels to numerical values
le = LabelEncoder()
le.fit(y_dx)
y = le.transform(y_dx)

# just for reference, the columns of the binarized label read respectively:
le.inverse_transform([0,1,2])

# <markdowncell>

# ## Label classification
# *Please note that the ROC plots will look different everytime the notebook is run due to the random nature of the cross-validation split*
# 
# ### Vanilla classifier
# We will go straight to using RandomForest and a 10-fold cross validation. Many other models were tried but RandomForest consistently prevented overfitting. First let's get an idea of how it looks like when you try to classify all the labels at the same time. 

# <codecell>

clf = RandomForestClassifier(n_estimators=50, oob_score=True)
clf.fit(X.values, y)
scores = cross_val_score(clf, X.values, y, cv=10)

print("Cross validation score:")
print(scores.mean())

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis = 0)
indices = np.argsort(importances)[::-1]

print("feature ranking:")

for f in range(20):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# <markdowncell>

# ### Build a One-vs-all ROC curve (not cross validated)
# To build a ROC curve we need to binarize the variable and run the classifier as one class vs. all others

# <codecell>

y_bin = label_binarize(y,classes=[0,1,2])
n_classes = y_bin.shape[1]

# <codecell>

X_train, X_test, y_train, y_test = train_test_split(X.values, y_bin, test_size=.3)

# <codecell>

clf1 = OneVsRestClassifier(RandomForestClassifier(n_estimators=50))
y_score = clf1.fit(X_train, y_train).predict_proba(X_test)

# <markdowncell>

# The probabilities of each class are now in a numpy array where each row corresponds to sample and each column to the label in question (CD, NM or UC). Let's take a pick at the first 10:

# <codecell>

y_score[:10,:]

# <codecell>

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = roc_auc_score(y_test[:,i], y_score[:,i],average="micro")

# <codecell>

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - UC vs all')
plt.legend(loc="lower right")
plt.show()

# <codecell>

# Plot ROC curves all together now
plt.figure()

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve {0} (area = {1:0.2f})'
                                   ''.format(le.inverse_transform(i), roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - one random 1/3 split')
plt.legend(loc="lower right")
plt.show()

# <markdowncell>

# ### ROC cross validated, one vs. all - Figure 6A

# <codecell>

# Run classifier with cross-validation and plot ROC curves
for dx in range(n_classes):
    cv = StratifiedKFold(y_bin[:,dx], n_folds=10)
    classifier = RandomForestClassifier()

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(cv):
        probas_ = classifier.fit(X.iloc[train], y_bin[train,dx]).predict_proba(X.iloc[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_bin[test,dx], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        #plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))


    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 
             label='Mean ROC %s (area = %0.2f)' % (le.inverse_transform(dx), mean_auc), lw=1)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Figure 6A - ROC - crossvalidated - one vs. all')
plt.legend(loc="lower right")
plt.show()

# <codecell>

# Only CD vs UC

X_cduc = X[(y == 0) | (y == 2)]
y_cduc = y[(y == 0) | (y == 2)]
np.place(y_cduc,y_cduc == 2, 1)

cv = StratifiedKFold(y_cduc, n_folds=10)
clf_cduc = RandomForestClassifier()

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    fitted = classifier.fit(X_cduc.iloc[train], y_cduc[train])
    probas_ = fitted.predict_proba(X_cduc.iloc[test])
    scored_ = fitted.predict(X_cduc.iloc[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_cduc[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    #roc_auc = auc(fpr, tpr)
    roc_auc = roc_auc_score(scored_, y_cduc[test], average="micro")
    #plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))


mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=1)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Figure 6A - ROC - crossvalidated - CD vs UC')
plt.legend(loc="lower right")
plt.show()

# <codecell>


# <codecell>


