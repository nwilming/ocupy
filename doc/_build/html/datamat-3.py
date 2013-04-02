import numpy as np
from ocupy import fixmat
import pylab as plot
points = np.random.random((2,100))*500
fm = fixmat.TestFixmatFactory(points = points, params = {'image_size' : [500,500]})
fdm = fixmat.compute_fdm(fm)
plot.imshow(fdm)
plot.show()