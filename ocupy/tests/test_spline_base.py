import unittest
import random
import numpy as np

from ocupy import measures, spline_base as sb 

from scipy.stats.kde import gaussian_kde
from scikits.learn import linear_model

class Test1DSplines(unittest.TestCase):

    def skip_cosine_fit(self):
        x = np.cos(np.linspace(0,np.pi*2, 99))
        noise = (np.random.random((99,))-.5)
        for num_splines in range(5,20,2):
            splines = sb.spline_base1d(99, num_splines, spline_order = 5)
            model =  linear_model.BayesianRidge()
            model.fit(splines, x+noise)
            y = model.predict(splines)
            self.assertTrue(np.corrcoef(x+noise, y)[0,1]**2>0.7)                     

class Test2DSplines(unittest.TestCase):
    
    def skip_2Dcosine_fit(self):
        x = np.cos(np.linspace(0,np.pi*2, 99))
        X = np.dot(x[:,np.newaxis],x[:,np.newaxis].T)
        noise = (np.random.random(X.shape)-.5)*0.1
        for num_splines in range(2,20,2):
            splines = sb.spline_base2d(99,99, num_splines, spline_order = 3)
            model =  linear_model.BayesianRidge()
            model.fit(splines.T, (X+noise).flat)
            y = model.predict(splines.T)
            self.assertTrue( np.corrcoef(X+noise, y.reshape(X.shape))[0,1]**2 > 0.7)

class TestCompareMethods(object):

    def setUp(self):
        # Generate fake density
        self.shape = (40,120)
        self.base = sb.spline_base2d(self.shape[0],self.shape[1], 20, 20, 10)  
        self.coef = np.random.random((self.base.shape[0],))
        self.target = np.dot(self.base.T, self.coef).reshape(self.shape)
        self.target += np.min(self.target.flat)
        self.target = self.target /np.sum(self.target)
        self.csum = np.cumsum(self.target.flat)
        
    def new_samples(self):
        self.train_samples = [np.unravel_index(draw_from(self.csum), self.shape) for x in range(5000)]
        xs = [x for x,y in self.train_samples]
        ys = [y for x,y in self.train_samples]
        self.train_samples = np.array((xs,ys)).T

    
    def test_compare(self):
        kls = []
        for k in range(5):
            self.new_samples()
            kls += self.eval_fits()
        print np.mean(np.array(kls))

    def eval_fits(self):
        print 'Fit Splines'
        spline_fit, hist = sb.fit2d(self.train_samples, 
                np.linspace(0,self.shape[0], self.shape[0]+1),
                np.linspace(0,self.shape[1], self.shape[1]+1))
        spline_kl = measures.kldiv(spline_fit, self.target)
        hist_kl = measures.kldiv(hist, self.target)
        try:
            print 'Fit KDE'
            kde_est = gaussian_kde(self.train_samples)
            x,y = np.mgrid[0:self.shape[0], self.shape[1]]
            kde_fit = kde_est.evaluate((np.array(x), np.array(y)))
            kde_kl = measures.kldiv(kde_fit, self.target)
        except:
           kde_kl = np.nan
           print 'LinAlgError' 
        print spline_kl, hist_kl, kde_kl
        return spline_kl, hist_kl, kde_kl
        
def r_squared(x,y):
    return (1 - np.linalg.norm(x - y)**2 / np.linalg.norm(x)**2)

def draw_from(cumsum, borders=None):
    """
    Draws a value from a cumulative sum.
    
    Parameters: 
        cumsum : array
            Cumulative sum from which shall be drawn.
        borders : array, optional
            If given, sets the value borders for entries in the cumsum-vector.
    Returns:
        int : Index of the cumulative sum element drawn.
    """
    if not borders is None:
        return (cumsum>=random.random()).nonzero()[0][0]
    else:
        return borders[(cumsum>=random.random()).nonzero()[0][0]]   

if __name__ == '__main__':
    test = TestCompareMethods()
    test.setUp()
    test.test_compare()
