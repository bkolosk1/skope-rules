import lightgbm as lgb
import shap
from sklearn.datasets import make_classification

# simulate raw data
X, y = make_classification(
    n_samples=1000,
    n_features=50,
    n_informative=5,
    n_classes=2,
    n_clusters_per_class=3,
    shuffle=False
)

# fit a GBT model to the data
m = lgb.LGBMClassifier()
m.fit(X, y)

# compute SHAP values
explainer = shap.Explainer(m)
shap_values = explainer(X)

from umap import UMAP

# compute 2D embedding of raw variable values
X_2d = UMAP(
  n_components=2, n_neighbors=200, min_dist=0
).fit_transform(X)

# compute 2D embedding of SHAP values
s_2d = UMAP(
  n_components=2, n_neighbors=200, min_dist=0
).fit_transform(shap_values.values)
from sklearn.cluster import DBSCAN

# Identify clusters using DBSCAN
s_labels = DBSCAN(eps=1.5, min_samples=20).fit(s_2d).labels_
import numpy as np
from skrules import SkopeRules

for cluster in np.unique(s_labels):
    # create target variable for individual cluster
    yc = (s_labels == cluster) * 1
    # use SkopeRules to identify rules with a maximum of two comparison terms
    sr = SkopeRules(max_depth=2).fit(X, yc)
    # print best decision rule
    print(cluster, sr.rules_[0][0])
    # print precision and recall of best decision rule
    print(f"Precision: {sr.rules_[0][1][0]:.2f}",
          f"Recall   : {sr.rules_[0][1][1]:.2f}")
    