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
    "gauss": lambda x, b: b.exp(-(x)),
    "inv": lambda x, b: 1.0 / (1.0 + x),
    "linear": lambda x, b: x,
    "logistic": lambda x, b: b.log(1 + b.exp(x)),
    "sigmoid": lambda x, b: 1/(1 + b.exp(-x)),
    "quadratic": lambda x, b: x ** 2,
    "LeReLU": lambda x, b: b.maximum(x, x*0.05),
}


def ffnn_predict(X,W, nn_name="nn", backend=np):
    """makes predictions with nn with """
    H = X

    idx = 0
    name = "w" # name for which to check if it is still in dict

    while (nn_name + "_w_%s"%idx) in W:
        w = W[nn_name + "_w_%s"%idx]
        b = W[nn_name + "_b_%s"%idx]
        a = W[nn_name + "_a_%s"%idx]
        H = backend.dot(H,w) + b
        H = activations[a](H, backend)
        idx += 1

    return H


def rnd_gen(*args):
    return np.random.randn(*args)


def select_nn(W, nn_name):
    R = {k:v for k, v in W.items() if k.startswith(nn_name)}
    return R



def make_ffnn_weights(X, Y, n_neurons, n_layers, act, out_act="linear", nn_name="nn", W={}, rnd = rnd_gen):
    """

    :param X: training inputs
    :param Y: training outputs
    :param n_neurons: number of neurons in feed forward nn
    :param n_layers: number of layers
    :param act: activation type
    :param out_act: activation applied to the output layer
    :param nn_name: name of the neural network
    :param W: dictionary where to write the results. New one is used if not provided.
    :return:
    """
    Xsz, Ysz = X.shape[-1], Y.shape[-1]
    Asz, Bsz = Xsz, n_neurons

    for idx in range(n_layers+1):

        if idx == n_layers:
            act = out_act
            Bsz = Ysz

        w = rnd(Asz, Bsz)
        b = rnd(Bsz)

        for name, par in (('w',w), ('b', b), ('a', act)):
            W[nn_name + "_" + name + "_%s"%idx] = par

        idx += 1
        Asz = Bsz

    return W
