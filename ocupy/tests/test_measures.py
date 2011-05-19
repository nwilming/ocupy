import unittest
import numpy as np
from ocupy import measures
from ocupy import fixmat
import scipy

class TestMeasures(unittest.TestCase):

    def test_prediction_scores(self):
        measures.set_scores([measures.roc_model, 
                             measures.kldiv_model, 
                             measures.nss_model])
        fm = fixmat.TestFixmatFactory(categories = [1,2,3], 
                filenumbers = [1,2,3,4,5,6],
                subjectindices = [1, 2, 3, 4, 5, 6],
                params = {'pixels_per_degree':10, 'image_size':[100,500]})
        fdm = fixmat.compute_fdm(fm)
        measures.set_scores([measures.roc_model, 
                             measures.kldiv_model, 
                             measures.nss_model])
        scores  =  measures.prediction_scores(fdm, fm) 
        self.assertEquals(len(scores), 3)
        measures.set_scores([measures.roc_model])
        scores  =  measures.prediction_scores(fdm, fm) 
        self.assertEquals(len(scores), 1)
        measures.set_scores([measures.roc_model, 
                             measures.kldiv_model, 
                             measures.nss_model])
        scores  =  measures.prediction_scores(fdm, fm) 
        self.assertEquals(len(scores), 3)

    def test_kldiv(self):
        arr = scipy.random.random((21,13))
        fm = fixmat.TestFixmatFactory(categories = [1,2,3], 
                filenumbers = [1,2,3,4,5,6],
                subjectindices = [1, 2, 3, 4, 5, 6],
                params = {'pixels_per_degree':10, 'image_size':[100,500]})
        
        kl = measures.kldiv(arr, arr)
        self.assertEqual(kl, 0, 
                "KL Divergence between same distribution should be 0")
        kl = measures.kldiv(None, None, distp = fm, distq = fm, scale_factor = 0.25)
        self.assertEqual(kl, 0, 
                "KL Divergence between same distribution should be 0")
        fdm = fixmat.compute_fdm(fm)
        kl = measures.kldiv_model(fdm, fm)
        self.assertTrue(kl < 10**-13, 
                "KL Divergence between same distribution should be almost 0")
        fm.x = []
        fm.y = []
        kl = measures.kldiv(None, None, distp = fm, distq = fm, scale_factor = 0.25)
        self.assertTrue(np.isnan(kl))

    def test_correlation(self):
        fm = fixmat.TestFixmatFactory(categories = [1,2,3], 
                filenumbers = [1,2,3,4,5,6],
                subjectindices = [1, 2, 3, 4, 5, 6],
                params = {'pixels_per_degree':1, 'image_size':[100,500]})
        # Arr has zero variance, should return nan
        arr = np.ones(fm.image_size)
        corr = measures.correlation_model(arr, fm)
        self.assertTrue(np.isnan(corr))
        # With itself should give 1
        fdm = fixmat.compute_fdm(fm)
        corr = measures.correlation_model(fdm,fm)
        self.assertEquals(corr,1)
        # Anti-correlation should give -1
        corr = measures.correlation_model(-1*fdm,fm)
        self.assertEquals(corr,-1)
    
    def test_nss_values(self):
        fm = fixmat.TestFixmatFactory(categories = [1,2,3], 
                filenumbers = [1,2,3,4,5,6],
                subjectindices = [1, 2, 3, 4, 5, 6],
                params = {'pixels_per_degree':0.1, 'image_size':[200,500]})
        # Arr has zero variance, should return nan
        arr = np.ones(fm.image_size)
        nss = measures.nss_model(arr, fm)
        self.assertTrue(np.isnan(nss))
        # With itself should yield a high value 
        fdm = fixmat.compute_fdm(fm)
        nss = measures.nss_model(fdm, fm)
        self.assertTrue(nss>15)
        # Fixations at these locations should give nss < 0
        nss = measures.nss(fdm, [[100, 101, 102, 103, 104, 105],[0, 0, 0, 0, 0, 0]])
        self.assertTrue(nss < 0)
    
    def test_emd(self):
       try: 
           import opencv
       except ImportError:
           print "Skipping EMD test - no opencv available"
           return 
       opencv # pyflakes
       fm = fixmat.TestFixmatFactory(categories = [1,2,3], 
           filenumbers = [1,2,3,4,5,6],
           subjectindices = [1, 2, 3, 4, 5, 6],
           params = {'pixels_per_degree':1, 'image_size':[20,50]})
       arr = np.ones(fm.image_size)
       fdm = fixmat.compute_fdm(fm)
       e = measures.emd_model(arr, fm)
       self.assertTrue(e > 0)
       e = measures.emd(fdm, fdm)
       self.assertEquals(e, 0) 

        
    def test_fast_roc(self):
        self.assertTrue(measures.fast_roc([1],[0])[0] == 1)
        self.assertTrue(measures.fast_roc([1],[1])[0] == 0.5)
        #self.assertTrue(measures.roc([0],[1])[0] == 0)
        self.assertTrue(np.isnan(measures.fast_roc([],[1])[0]))
        self.assertTrue(np.isnan(measures.fast_roc([1],[])[0]))
        self.assertRaises(RuntimeError, lambda: measures.fast_roc([np.nan],[0]))
        self.assertRaises(RuntimeError, lambda: measures.fast_roc([0],[np.nan]))

        # test auc linearity
        actuals = np.random.standard_normal(1000)+2
        controls = np.random.standard_normal(1000)
        auc_complete = measures.fast_roc(actuals, controls)[0]
        auc_partial = [measures.fast_roc(actuals[k*100:(k+1)*100],controls)[0] for k in range(10)]
        self.assertAlmostEqual(auc_complete,np.array(auc_partial).mean())

        # test symmetry
        actuals = np.random.standard_normal(1000)+2
        controls = np.random.standard_normal(1000)
        self.assertAlmostEqual(measures.fast_roc(actuals, controls)[0] + measures.fast_roc(controls, actuals)[0],1)


    def test_exact_roc(self):
        self.assertTrue(measures.exact_roc([1],[0])[0] == 1)
        self.assertTrue(measures.exact_roc([1],[1])[0] == 0.5)
        self.assertTrue(measures.exact_roc([0],[1])[0] == 0)
        self.assertTrue(np.isnan(measures.exact_roc([],[1])[0]))
        self.assertTrue(np.isnan(measures.exact_roc([1],[])[0]))
        self.assertRaises(RuntimeError, lambda: measures.exact_roc([np.nan],[0]))
        self.assertRaises(RuntimeError, lambda: measures.exact_roc([0],[np.nan]))
        # test auc linearity
        actuals = np.random.standard_normal(1000)+2
        controls = np.random.standard_normal(1000)
        auc_complete = measures.exact_roc(actuals, controls)[0]
        auc_partial = [measures.exact_roc(actuals[k*100:(k+1)*100],controls)[0] for k in range(10)]
        self.assertAlmostEqual(auc_complete,np.array(auc_partial).mean())

        # test symmetry
        actuals = np.random.standard_normal(1000)+2
        controls = np.random.standard_normal(1000)
        self.assertAlmostEqual(measures.exact_roc(actuals, controls)[0] + measures.exact_roc(controls, actuals)[0],1)

    def skip_intersubject_auc(self):
        points = zip(range(1,16),range(1,16))
        fm = fixmat.TestFactory(points = points, 
            filenumbers = range(1,11), subjectindices = range(1,11)) 
        (auc1, _, _) = measures.intersubject_scores_random_subjects(fm, 1, 1, 2, 2, False)
        (auc2, _, _) = measures.intersubject_scores_random_subjects(fm, 1, 1, 2, 2, True)

    def skip_intersubject_auc_scaled(self):
        points = zip(range(1,16),range(1,16))
        fm = fixmat.TestFactory(points = points, 
            filenumbers = range(1,11), subjectindices = range(1,11)) 
        (auc1, _, _) = measures.intersubject_scores(fm, 1,[1], [1], [2], [2], controls=False, scale_factor = 0.5)
        (auc2, _, _) = measures.intersubject_scores(fm,1, [1], [1], [2],[2], controls=True, scale_factor = 0.5)
   
 
    def test_nss(self):
        fm = fixmat.TestFixmatFactory(points=zip([0,500,1000],[1,10,10]),params = {'image_size':[100,10]})
        fm.SUBJECTINDEX = np.array([1,1,1])
        fm.filenumber = np.array([1,1,1])
        fm.category = np.array([1,1,1])
        fm.x = np.array([0,50,1000])
        fm.y = np.array([1,10,10])
        fm.fix = np.array([1,2,3])
        fdm = fixmat.compute_fdm(fm[(fm.x<10) & (fm.y<10)])
        self.assertRaises(IndexError, lambda: measures.nss(fdm, (fm.y, fm.x))) 

    def skip_emd(self):
        fm1 = fixmat.TestFactory(params = {'image_size':[93,128]})
        fm2 = fixmat.TestFactory(points=zip(range(10,50),range(10,50)),params = {'image_size':[93,128]})
        self.assertEquals(measures.emd_model(fixmat.compute_fdm(fm1), fm1), 0)
        self.assertTrue(not (measures.emd_model(fixmat.compute_fdm(fm1), fm2) == 0))


if  __name__ == '__main__':
   unittest.main()
