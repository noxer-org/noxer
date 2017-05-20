# Noxer

Streamlined supervised learning for easy training and deployment of ML models.

## Installation

Install using python package manager in terminal

* For users: `[sudo] pip install https://github.com/iaroslav-ai/noxer/archive/master.zip`
* For contributors:
```python
git clone https://github.com/iaroslav-ai/noxer.git
cd noxer
[sudo] pip install -e .
```

## Minimal example

Sequence classification example is shown below.
Sequence padding is used as a preprocessing step,
and Recurrent Neural Network as an estimator:

```python
import noxer.sequences as nx

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

import numpy as np
import cPickle as pc

# a set of annotated sequences of same size
X = np.random.randn(2 ** 10,16,2) # [:, seq_len, features]
y = X[:,-1,0] > 0.0 # can array of objects

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)

# create preprocessing and estimation pipeline
model = make_pipeline(
    nx.PaddedSubsequence(length=2), # Select last 2 time sequence elements in all input sequences
    nx.RNNClassifier(n_neurons=32) # Apply recurrent neural network
)

# train and evaluate model
model.fit(X, y)
print model.score(X_test, y_test)

# save the model
with open("model.bin", "w") as f:
    pc.dump(model, f)

# load the model and do estimations
with open("model.bin", "r") as f:
    model = pc.load(f)

print model.predict(X_test)
```

Other classes from sklearn such as `GridSearchCV` can be used with pipelines for hyperparameter search.