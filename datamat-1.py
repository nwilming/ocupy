import numpy as np
from ocupy import fixmat
import pylab as plot
fm = fixmat.FixmatFactory('../ocupy/tests/datamat_demo.mat')
fdm = fixmat.compute_fdm(fm,scale_factor=0.25)
plot.imshow(fdm)
plot.show()