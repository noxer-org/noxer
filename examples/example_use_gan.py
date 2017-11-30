import torch
import numpy as np
import matplotlib.cm as cm

name = '/home/iaroslav/datasets/fashion.net'
model = torch.load(open(name, 'rb'))

xp = np.array(list(range(0, 10))*10)
yp = model.predict(xp)
print(yp.shape)
import matplotlib.pyplot as plt
import math

W = math.ceil(math.sqrt(len(yp)))

map = {0: "T-shirt/top",
1: "Trouser",
2: "Pullover",
3: "Dress",
4: "Coat",
5: "Sandal",
6: "Shirt",
7: "Sneaker",
8: "Bag",
9: "Ankle boot"}

for i, (v, xv) in enumerate(zip(yp, xp)):
    plt.subplot(W, W, i + 1)
    if v.shape[-1] == 1:
        v = v[:, :, 0]

    plt.title(map[xv])
    plt.imshow(v, cmap = cm.Greys_r)
    plt.axis('off')

plt.show()