import numpy as np
from ocupy import fixmat
import pylab as plot
fm = fixmat.FixmatFactory('../ocupy/test/datamat_demo.mat')
fm = fm[(fm.filenumber == 1) & (fm.category == 7)]
fdm = fixmat.compute_fdm(fm,scale_factor=0.25)
plot.imshow(fdm)
plot.show()