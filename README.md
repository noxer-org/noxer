[![CircleCI](https://circleci.com/gh/noxer-org/noxer.svg?style=svg)](https://circleci.com/gh/noxer-org/noxer)
[![Code Health](https://landscape.io/github/noxer-org/noxer/master/landscape.svg?style=flat)](https://landscape.io/github/noxer-org/noxer/master)
![Python 3.5](https://img.shields.io/badge/python-3.5-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Noxer

This package aims to simplify working with a range of AI problems. We keep
 interfaces of our code as simple as possible while maintaining reasonable flexibility 
 for the extensions of the code.

At current, the following is possible:
1. Supervised learning with deep, recurrent convoneural neural networks with PyTorch backend.
2. Learning generative models such as VAE and GAN's for distributions simple and complex.

## What are the benefits of using Noxer

We reuse a lot `scikit-learn` like interfaces. This yields a few benefits:
* No learning curve to use the code. In fact, you can use all the models same as you
do it in sklearn. 
* Efficient preprocessing of the data for datasets that fit in memory using `FeatureUnion`
or `Pipeline` classes. One could possibly go far beyond memory size with `dask`.
* All machinery necessary for hyperparameter setting and selection. The code can be used 
directly with GridSearchCV from `scikit-learn` or better yet with BayesSearchCV 
from `scikit-optimize` that is more efficient in number of model trainings. 

We utilize a recent PyTorch deep learning framework. This leads to a few benefits also:
* There is no compilation time of a neural network - nice feature for impatient developers
* Debugging can be done similar to debugging of any other Python code
* Serialization and deserialization is as easy as `torch.save(...)` and
 `torch.load(...)`, which is an extension of pickle (and hence works with any
 objects).


## Installation

Install using pip in terminal:

* If you only want to use code: `[sudo] pip install https://github.com/iaroslav-ai/noxer/archive/master.zip`
* If you want to edit the code:
```bash
git clone https://github.com/iaroslav-ai/noxer.git
cd noxer
sudo pip install -e .
```

## Examples

See example usage in `examples` folder. 

## Warning

This package is actively developed, but is still in beta. Do expect the unexpected!

Icon made by Freepik from www.flaticon.com .

## For most part this software is ...

![under construction.](https://iaroslav-ai.github.io/images/under_construction.svg)