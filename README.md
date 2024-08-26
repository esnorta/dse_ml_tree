# Intro
The goal of this project is the implementation of the tree predictors for binary classification from scratch in Python3. 
Tree predictor fitting and evaluation is performed using the Mushroom dataset. 

# Modules Description
The core code contains several modules responsible for different procedures of tree predictor fitting and evaluation:
- tree.py
The Tree Class implements methods for the construction of the tree and predicting the examples’ labels.
- splitter.py
The Splitter Class is responsible for dataset splitting: finding best split, train/test, k-folds generating.
- entropy.py, impurity.py
Contain classes for entropy and impurity estimation of the data provided.
- evaluator.py
Evaluator implements estimation of the model performance metrics: accuracy, precision and recall.

# Tree Predictor Features
- Tree has a root initialised at start; after the root the child nodes are created recursively if split condition is found.
- Single-feature binary tests are used in each node to asses potential splits.
- For selecting the best split a splitting criterion is used: Entropy, Gini impurity or Scaled impurity.
- Two split conditions are implemented: Threshold and In-set (membership).
- On each split a stopping criterion is controlled: minimum gain or maximum depth. If the stopping condition is reached the node is converted to a leaf with target label probabilities.

# Installation
`pip3 install -r requirements.txt`

# Usage
See the Jupyter Notebook.

# Tests
`python3 -m pytest -s`
