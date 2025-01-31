"""
============================================
Example: Detecting Defaults on Retail Credits
============================================

SkopeRules finds logical rules with high precision and fuses them. Finding
good rules is done by fitting classification and regression trees
to sub-samples. A fitted tree defines a set of rules (each tree node defines a rule);
rules are then tested out of the bag, and the ones with higher precision are kept.

This example aims at finding logical rules to predict credit defaults.
The analysis shows that setting appropriate precision and recall thresholds
allows for the extraction of interpretable rules that perform comparably to
a Random Forest classifier.

###############################################################################
# Data Import and Preparation
# ----------------------------
#
# There are 3 categorical variables (SEX, EDUCATION, and MARRIAGE) and 20
# numerical variables. The target (credit defaults) is transformed into a binary
# variable with integers 0 (no default) and 1 (default). From the 30,000 credits,
# 50% are used for training and 50% are used for testing. The target is unbalanced
# with a 22%/78% ratio.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

# Import the updated SkopeRules class
from skrules import SkopeRules
from skrules.datasets import load_credit_data

print(__doc__)

# Set random seed for reproducibility
rng = np.random.RandomState(1)

# Importing data
dataset = load_credit_data()
X = dataset.data
y = dataset.target

# Shuffling data, preparing target and variables
X_shuffled, y_shuffled = shuffle(np.array(X), y, random_state=rng)
data = pd.DataFrame(X_shuffled, columns=dataset.feature_names)

# Remove the 'ID' column if it exists
if 'ID' in data.columns:
    data.drop(['ID'], axis=1, inplace=True)

# Quick feature engineering
data = data.rename(columns={"PAY_0": "PAY_1"})

# Calculate mean for old PAY columns
old_PAY = ['PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
data['PAY_old_mean'] = data[old_PAY].mean(axis=1)

# Calculate mean and std for old BILL_AMT columns
old_BILL_AMT = ['BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
data['BILL_AMT_old_mean'] = data[old_BILL_AMT].mean(axis=1)
data['BILL_AMT_old_std'] = data[old_BILL_AMT].std(axis=1)

# Calculate mean and std for old PAY_AMT columns
old_PAY_AMT = ['PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
data['PAY_AMT_old_mean'] = data[old_PAY_AMT].mean(axis=1)
data['PAY_AMT_old_std'] = data[old_PAY_AMT].std(axis=1)

# Drop the original old columns
data.drop(old_PAY_AMT + old_BILL_AMT + old_PAY, axis=1, inplace=True)

# Creating the train/test split
feature_names = list(data.columns)
print("List of variables used to train models:", feature_names)

data_values = data.values
n_samples = data_values.shape[0]
n_samples_train = n_samples // 2  # Integer division for training samples

# Split the data
y_train = y_shuffled[:n_samples_train]
y_test = y_shuffled[n_samples_train:]
X_train = data_values[:n_samples_train]
X_test = data_values[n_samples_train:]

###############################################################################
# Benchmark with a Random Forest Classifier
# -----------------------------------------
#
# This part shows the training and performance evaluation of a Random Forest
# model. The objective remains to extract rules which target credit defaults.

# Define the parameter grid for GridSearchCV
param_grid = {
    'max_depth': range(3, 8, 1),
    'max_features': np.linspace(0.1, 1.0, 5)
}

# Initialize GridSearchCV with RandomForestClassifier
rf = GridSearchCV(
    estimator=RandomForestClassifier(
        random_state=rng,
        n_estimators=50,
        class_weight='balanced'
    ),
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    refit=True,
    n_jobs=-1
)

# Fit the Random Forest model
rf.fit(X_train, y_train)

# Predict probabilities for the test set
scoring_rf = rf.predict_proba(X_test)[:, 1]

print("Random Forest selected parameters:", rf.best_params_)

# Plot ROC and Precision-Recall curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

# ROC Curve for Random Forest
ax = axes[0]
fpr_RF, tpr_RF, _ = roc_curve(y_test, scoring_rf)
ax.step(fpr_RF, tpr_RF, linestyle='-.', color='g', lw=1, where='post', label="Random Forest")
ax.set_title("ROC", fontsize=20)
ax.set_xlabel('False Positive Rate', fontsize=18)
ax.set_ylabel('True Positive Rate (Recall)', fontsize=18)
ax.legend(loc='lower right', fontsize=12)

# Precision-Recall Curve for Random Forest
ax = axes[1]
precision_RF, recall_RF, _ = precision_recall_curve(y_test, scoring_rf)
ax.step(recall_RF, precision_RF, linestyle='-.', color='g', lw=1, where='post', label="Random Forest")
ax.set_title("Precision-Recall", fontsize=20)
ax.set_xlabel('Recall (True Positive Rate)', fontsize=18)
ax.set_ylabel('Precision', fontsize=18)
ax.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.show()

###############################################################################
# The ROC and Precision-Recall curves illustrate the performance of Random
# Forests in this classification task. Suppose now that we add an interpretability
# constraint to this setting:
# Typically, we want to express our model in terms of logical rules detecting
# defaults. A Random Forest could be expressed in terms of a weighted sum of
# rules, but:
# 1) Such a large weighted sum is hardly interpretable.
# 2) Simplifying it by removing rules/weights is not easy, as optimality is
#    targeted by the ensemble of weighted rules, not by each rule.
# In the following section, we show how SkopeRules can be used to produce
# a number of rules, each seeking high precision on a potentially small
# area of detection (low recall).

###############################################################################
# Getting Rules with SkopeRules
# ------------------------------
#
# This part shows how SkopeRules can be fitted to detect credit defaults.
# Performances are compared with the Random Forest model previously trained.

# Initialize SkopeRules with desired parameters
clf = SkopeRules(
    max_depth_duplication=3,
    max_depth=3,
    max_features=0.5,
    max_samples_features=0.5,
    random_state=rng,
    n_estimators=20,
    feature_names=feature_names,
    recall_min=0.04,
    precision_min=0.6
)

# Fit the SkopeRules model
clf.fit(X_train, y_train)

# In the score_top_rules method, a score of k means that rule number k
# votes positively, but not rules 1, ..., k-1. It allows us to plot
# the performance of each rule separately on the ROC and PR plots.
scoring = clf.score_top_rules(X_test)

print(f"{len(clf.rules_)} rules have been built.")
print("The 5 most precise rules are the following:")
for rule in clf.rules_[:5]:
    print(rule[0])

# Define the curves to plot
curves = [roc_curve, precision_recall_curve]
xlabels = ['False Positive Rate', 'Recall (True Positive Rate)']
ylabels = ['True Positive Rate (Recall)', 'Precision']

# Plot ROC and Precision-Recall curves for SkopeRules vs Random Forest
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

# ROC Curve
ax = axes[0]
fpr, tpr, _ = roc_curve(y_test, scoring)
fpr_rf, tpr_rf, _ = roc_curve(y_test, scoring_rf)
ax.scatter(fpr[:-1], tpr[:-1], c='b', s=10, label="Rules of SkopeRules")
ax.step(fpr_RF, tpr_RF, linestyle='-.', color='g', lw=1, where='post', label="Random Forest")
ax.set_title("ROC", fontsize=20)
ax.set_xlabel('False Positive Rate', fontsize=18)
ax.set_ylabel('True Positive Rate (Recall)', fontsize=18)
ax.legend(loc='lower right', fontsize=12)

# Precision-Recall Curve
ax = axes[1]
precision, recall, _ = precision_recall_curve(y_test, scoring)
precision_rf, recall_rf, _ = precision_recall_curve(y_test, scoring_rf)
ax.scatter(recall[:-1], precision[:-1], c='b', s=10, label="Rules of SkopeRules")
ax.step(recall_RF, precision_RF, linestyle='-.', color='g', lw=1, where='post', label="Random Forest")
ax.set_title("Precision-Recall", fontsize=20)
ax.set_xlabel('Recall (True Positive Rate)', fontsize=18)
ax.set_ylabel('Precision', fontsize=18)
ax.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.show()

###############################################################################
# The ROC and Precision-Recall curves show the performance of the rules
# generated by SkopeRules (the blue points) and the performance of the
# Random Forest classifier fitted above.
# Each blue point represents the performance of a set of rules:
# Starting from the left on the precision-recall curve, the k-th point
# represents the score associated with the concatenation (union) of the first k rules, etc.
# Thus, each blue point is associated with an interpretable classifier,
# which is a combination of a few rules.
# In terms of performance, each of these interpretable classifiers compares well
# with Random Forest, while offering complete interpretation.
# The range of recall and precision can be controlled by the precision_min and
# recall_min parameters. Here, setting precision_min to 0.6 forces the rules to
# have a limited recall.
