# load the data
import numpy as np
import cloudpickle as pc
import torch

Y, X = pc.load(open('mnist.bin', 'rb'))[0]
Y = Y[:, :, :, np.newaxis] / 256.0

from noxer.gm.gan import ACGANCategoryToImageGenerator

model = ACGANCategoryToImageGenerator(
    verbose=1,
    epochs=32
)

def callback():
    xp = X[:16]
    yp = model.predict(xp)
    print(yp.shape)
    import matplotlib.pyplot as plt
    plt.ion()
    import math

    W = math.ceil(math.sqrt(len(yp)))

    for i, (v, xv) in enumerate(zip(yp, xp)):
        plt.subplot(W, W, i + 1)
        if v.shape[-1] == 1:
            v = v[:, :, 0]

        plt.title(xv)
        plt.imshow(v)

    plt.show()
    plt.pause(0.1)

model.fit(X, Y, callback=callback)
torch.save(model, open('m.bin', 'wb'), pickle_module=pc)