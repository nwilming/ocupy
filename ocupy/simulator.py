from tools import *
from ocupy import fixmat
from math import asin, atan2, degrees, radians
import spline_base
from progressbar import ProgressBar, Percentage, Bar
import numpy as np
#import pycallgraph

class AbstractSim(object):
	def __init__(self):
		raise NotImplementedError
	def sample(self):
		raise NotImplementedError
	def parameters(self):
		raise NotImplementedError
	def finish(self):
		raise NotImplementedError


class FixSim(AbstractSim):
	def __init__(self, fm_name = '/Users/nwilming/Documents/WML/analysis/qoamp/fixmat_photos.mat'):
		if type(fm_name)==str:
			self.fm = fixmat.FixmatFactory(fm_name)
		else:
			self.fm = fm_name
		
		if (min(self.fm.fix)==1):
			self.firstfixcentered = False
		else:
			self.firstfixcentered = True
			
		
	def initializeData(self):
		ad, ld = anglendiff(self.fm, roll=1) # Calculate diffs for the fixmat
		screen_diag = int(ceil((np.sum(np.array(self.fm.image_size)**2)**.5)/self.fm.pixels_per_degree))
		
		self.full_H1 = spline(ad[0],ld[0]/self.fm.pixels_per_degree,collapse=False,xdim=[-screen_diag,screen_diag])

		self.bloatfactor = 1
		
		self.firstLengthsAngles_cumsum, self.firstLengthsAngles_shape = (
									compute_cumsum(self.fm, 'la'))
		self.probability_cumsum = np.cumsum(np.concatenate(self.full_H1))
		self.firstcoordinates_cumsum = compute_cumsum(self.fm,'coo')
		self.trajectoryLengths_cumsum, self.trajectoryLengths_borders = compute_cumsum(self.fm, 'len')
		self.canceled = 0
		self.minusSaccades = 0
		
		
	def _calc_xy(self, (x,y), angle, length):
		return (x+(cos(radians(angle))*length),
				y+(sin(radians(angle))*length))
			   
	def _draw(self, prev_angle = None, prev_length = None):

		if (prev_angle == None) or (prev_length == None):
			(length, angle) = np.unravel_index(drawFrom(self.firstLengthsAngles_cumsum),
					self.firstLengthsAngles_shape)
			angle = angle-((self.firstLengthsAngles_shape[1]-1)/2.0)
			angle = 0
			#length = 100000000 #delete this
		else:
			J,I = np.unravel_index(drawFrom(self.probability_cumsum), self.full_H1.shape)
			
			angle = reshift((I-self.full_H1.shape[1]/2.0)/self.bloatfactor + prev_angle)
			
			ang_diff = (I-self.full_H1.shape[1]/2.0)/self.bloatfactor

			length = prev_length + ((J-self.full_H1.shape[0]/2.0)*self.fm.pixels_per_degree/self.bloatfactor)
		return angle, length
	
	def parameters(self):
		return {'fixmat':self.fm, 'sampling_dist':self.full_H1}

	def finish(self):
		pass
	
	def sample_many(self, num_samples = 100):
		#pycallgraph.start_trace()
		x = []
		y = []
		fix = []
		sample = []
		
		print "Simulating "+repr(num_samples)+" trajectories..."
		pbar = ProgressBar(widgets=[Percentage(),Bar()],maxval=num_samples).start()
		
		for s in xrange(0, num_samples):
			for i, (xs,ys) in enumerate(self.sample()):
				x.append(xs)
				y.append(ys)
				fix.append(i+1)
				sample.append(s)   
			
			pbar.update(s+1)
		fields = {'fix':np.array(fix),'y':np.array(y), 'x':np.array(x)}
		param = {'image_size':self.fm.image_size,'pixels_per_degree':self.fm.pixels_per_degree}
		out =  fixmat.VectorFixmatFactory(fields, param)
		pbar.finish()
		print repr(round((float(self.canceled)+self.minusSaccades)/len(out.x),3)*100) + "% canceled, of which "+repr(round(float(self.minusSaccades)/len(out.x),3)*100)+"% are minusSaccades."
		#pycallgraph.make_dot_graph('test.png')
		ad,ld = anglendiff(out,roll=2)
		H = spline(ad[0],ld[0]/out.pixels_per_degree)
		print "First spline done"
		H2 = spline(ad[1],ld[1]/out.pixels_per_degree)
		return out,H,H2#, self.canceled, self.minusSaccades
		
	def sample(self):
		lenghts = []
		angles = []
		coordinates = []
		fix = []
		sample_size = int(round(drawFrom(self.trajectoryLengths_cumsum, borders=self.trajectoryLengths_borders)))
		#sample_size = 5
		
		if (self.firstfixcentered == True):
			coordinates.append([self.fm.image_size[1]/2,self.fm.image_size[0]/2])
		else:
			K,L=(np.unravel_index(drawFrom(self.firstcoordinates_cumsum),[self.fm.image_size[0],self.fm.image_size[1]]))
			coordinates.append([L,K])
			
		fix.append(1)
		while len(coordinates) < sample_size:
			if len(lenghts) == 0 and len(angles) == 0:			
				angle, length = self._draw(self)													   
			else:
				angle, length = self._draw(prev_angle = angles[-1], prev_length = lenghts[-1])  
						
			x, y = self._calc_xy(coordinates[-1], angle, length) 
			
			if (False): #x<0 or x>self.fm.image_size[1] or y<0 or y>self.fm.image_size[0]: #(False):
				self.canceled+=1
				pass # Drawn coordinates were out of bounds
			elif (length<0):
				self.minusSaccades+=1
				pass # Drawn saccade length not possible
			else:
				coordinates.append([x,y])
				lenghts.append(length) 
				angles.append(angle)
				fix.append(fix[-1]+1)
		return coordinates

if __name__ == '__main__':
	sim = FixSim('fixmat_photos.mat')
	#sim.set_path()
	simfm = sim.sample_many(num_samples=6263)
