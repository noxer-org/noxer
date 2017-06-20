"""
Various reusable manipulations on shapes.

"""

import numpy as np

def struct_to_vector(S):
    """Converts an arbitrary dictionary of shapes into a single vector"""

    blueprint = []
    vect = []

    for k,v in S.items():
        if not isinstance(v, np.ndarray):
            blueprint.append((k, "misc", v))
            continue
        vf = v.flatten()
        st = len(vect)
        ed = st + len(vf)
        vect.extend(vf)
        meta_data = (k, "value", (slice(st, ed), v.shape))
        blueprint.append(meta_data)

    return np.array(vect), blueprint


def vector_to_struct(vect, blueprint):
    """Inverse to above function."""
    S = {}
    for k, t, v in blueprint:
        if t == "misc":
            S[k] = v
            continue

        slc, shape = v
        S[k] = np.reshape(vect[slc], newshape=shape)
    return S


activations = {
    "gauss": lambda x: np.exp(-(x)),
    "inv": lambda x: 1.0 / (1.0 + x),
    "linear": lambda x: x,
    "logistic": lambda x: np.log(1 + np.exp(x)),
    "sigmoid": lambda x: 1/(1 + np.exp(-x)),
    "quadratic": lambda x: x ** 2,
    "LeReLU": lambda x: np.maximum(x, x*0.05),
}


def ffnn_predict(X,W):
    """makes predictions with nn with """
    H = X
    idx = 0

    while ("w_%s"%idx) in W:
        w = W["w_%s"%idx]
        b = W["b_%s"%idx]
        a = W["a_%s"%idx]
        H = np.dot(H,w) + b
        H = activations[a](H)
        idx += 1

    return H


def make_ffnn_weights(X, Y, n_neurons, n_layers, act, out_act="linear"):
    Xsz, Ysz = X.shape[-1], Y.shape[-1]
    Asz, Bsz = Xsz, n_neurons

    W = {}

    for idx in range(n_layers+1):

        if idx == n_layers:
            act = out_act
            Bsz = Ysz

        w = np.random.randn(Asz, Bsz)
        b = np.random.randn(Bsz)
        W["w_%s"%idx] = w
        W["b_%s"%idx] = b
        W["a_%s"%idx] = act
        idx += 1
        Asz = Bsz

    return W
