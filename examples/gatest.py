# load the data
import numpy as np

X = np.random.randint(0, 10, size=10000)
Y = np.zeros((10000, 28, 28, 1))

from noxer.gm.gan import ACGANCategoryToImageGenerator

model = ACGANCategoryToImageGenerator(verbose=1)
model.fit(X, Y)