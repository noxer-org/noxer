"""Trains a recurrent neural network classifier on the random data, saves
the data preprocessing and model pipeline to the file, loads the pipeline
and makes estimation with with loaded pipeline.
A minimal example for explanatory purposes only.
"""

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
    nx.PadSubsequence(length=2), # Select last 2 time sequence elements in all input sequences
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