
import unittest as unittest

	
import mesaPlot as mp
import os
import matplotlib.pyplot as plt

os.chdir('tests')

class TestFileReader(unittest.TestCase):
	def test_init(self):
		m=mp.MESA()

	def test_load_history1(self):
		m=mp.MESA()
		m.loadHistory()
		
	def test_load_history2(self):
		m=mp.MESA()
		m.log_fold='LOGS/'
		m.loadHistory()	

	def test_load_profile1(self):
		m=mp.MESA()
		m.loadProfile(num=1)
		
	def test_load_profile2(self):
		m=mp.MESA()
		m.log_fold='LOGS/'
		m.loadProfile(num=-1)

	def test_load_profile3(self):
		m=mp.MESA()
		m.log_fold='LOGS/'
		m.loadProfile(num=-2)
		
	def test_load_profile3(self):
		m=mp.MESA()
		m.log_fold='LOGS/'
		m.loadProfile(f='LOGS/profile1.data')
		
		
class TestPlot(unittest.TestCase):
	def setUp(self):
		self.m=mp.MESA()
		self.p=mp.plot()
		self.m.loadHistory()
		self.m.loadProfile(num=1)
		
	def tearDown(self):
		plt.close('all')
		
	def test_plotHR(self):
		self.p.plotHR(self.m,show=False)
		
	def test_plotNeu(self):
		self.p.plotNeu(self.m,show=False)
		
	def test_abun(self):
		self.p.plotAbun(self.m,show=False)
		
	def test_plotAbunByA(self):
		self.p.plotAbunByA(self.m,show=False)
		
	def test_plotAbunHist(self):
		self.p.plotAbunHist(self.m,show=False)
		
	def test_plotAbunPAndN(self):
		self.p.plotAbunPAndN(self.m,show=False)
		
	def test_plotAngMom(self):
		self.p.plotAngMom(self.m,show=False)
		
	def test_plotBurn(self):
		self.p.plotBurn(self.m,show=False)
		
	def test_plotBurnHist(self):
		self.p.plotBurnHist(self.m,show=False)
		
	def test_plotDynamo(self):
		self.p.plotDynamo(self.m,show=False)
		
	def test_plotHistory(self):
		self.p.plotHistory(self.m,show=False)
		
	def test_plotProfile(self):
		self.p.plotProfile(self.m,show=False)
		
	def test_plotKip(self):
		self.p.plotKip(self.m,show=False)

	def test_plotKip2(self):
		self.p.plotKip2(self.m,show=False)
		
	def test_plotKip3(self):
		self.p.plotKip3(self.m,show=False)
		
	def test_plotKip3(self):
		self.p.plotKip3(self.m,show=False,age_lookback=True,xaxis='star_age')
	
	def test_plotMix(self):
		self.p.plotMix(self.m,show=False)	
		
	def test_plotTRho(self):
		self.p.plotTRho(self.m,show=False)
                
	def test_plotLdivM(self):
		self.p.plotLdivM(self.m,show=False)
		
