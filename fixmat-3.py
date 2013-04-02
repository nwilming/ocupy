import numpy as np
from ocupy import fixmat
import pylab as plot
fm = fixmat.FixmatFactory('../ocupy/tests/fixmat_demo.mat')
fm = fm[(fm.filenumber > 36) & (fm.filenumber < 40)]
for cat,(cat_mat,_) in enumerate(fm.by_cat()):
        for img,(img_mat,_) in enumerate(cat_mat.by_filenumber()):
                fdm = fixmat.compute_fdm(img_mat,scale_factor=0.25)
                plot.subplot(2,3, (3*cat)+img)
                plot.xticks([])
                plot.yticks([])
                plot.imshow(fdm)
plot.show()