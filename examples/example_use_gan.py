import torch
import numpy as np

name = 'fashion.net'
model = torch.load(open(name, 'rb'))

xp = np.array(list(range(0, 10))*10)
yp = model.predict(xp)
print(yp.shape)
import matplotlib.pyplot as plt
import math

W = math.ceil(math.sqrt(len(yp)))

for i, (v, xv) in enumerate(zip(yp, xp)):
    plt.subplot(W, W, i + 1)
    if v.shape[-1] == 1:
        v = v[:, :, 0]

    plt.title(xv)
    plt.imshow(v)

plt.show()