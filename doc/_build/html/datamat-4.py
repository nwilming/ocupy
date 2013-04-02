import numpy as np
from ocupy import datamat
import pylab as plot
points = np.random.random((2,100))*500
fm = datamat.TestFixmatFactory(points = points, params = {'image_size' : [500,500]})
fdm = datamat.compute_fdm(fm)
plot.imshow(fdm)
plot.show()