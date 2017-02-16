# Copyright (c) 2015, Robert Farmer rjfarmer@asu.edu

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


from __future__ import print_function
import numpy as np
import mmap
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import bisect
import scipy.interpolate as interpolate
from matplotlib.ticker import MaxNLocator,AutoMinorLocator
import matplotlib.patheffects as path_effects
import os
import random
import glob
import subprocess
from io import BytesIO
from cycler import cycler
from scipy.interpolate import interp1d

try:
   #Can be a problem on mac's
   mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
except:
   pass
## for Palatino and other serif fonts use:
#mpl.rc('font',**{'family':'serif','serif':['Palatino']})   


try:
   #Again can be problematic on mac's
   #TODO: create a flag and fix labels
   mpl.rc('text', usetex=True)
   x=r'$log_{10}$'
except:
   mpl.rc('text', usetex=False)



mpl.rc('font',size=32)
mpl.rc('xtick', labelsize=28) 
mpl.rc('ytick', labelsize=28) 
mpl.rcParams['axes.linewidth'] = 2.0
mpl.rcParams['xtick.major.size']=18      # major tick size in points
mpl.rcParams['xtick.minor.size']=9      # minor tick size in points
mpl.rcParams['ytick.major.size']=18      # major tick size in points
mpl.rcParams['ytick.minor.size']=9      # minor tick size in points

mpl.rcParams['xtick.major.width']=0.8      # major tick size in points
mpl.rcParams['xtick.minor.width']=0.6      # minor tick size in points
mpl.rcParams['ytick.major.width']=0.8      # major tick size in points
mpl.rcParams['ytick.minor.width']=0.6      # minor tick size in points

class data(object):
	def __init__(self):
		self.data={}
		self.head={}
		self._loaded=False
		

	def __getattr__(self, name):
		x=None
		
		if '_loaded' in self.__dict__:
			if self._loaded:
				try:
					x=self.data[name]
				except:
					pass
				try:
					x=np.atleast_1d(self.head[name])[0]
				except:
					pass
				
				if x is not None:
					return x
				else:
					raise AttributeError("No value "+name+" available")
						
		raise AttributeError("Must call loadHistory or loadProfile first")
	
	def __dir__(self):
		x=[]
		try:
			if len(self.head_names)>0:
				x=x+list(self.head_names)
		except:
			pass
		try:
			if len(self.data_names)>0:
				x=x+list(self.data_names)
		except:
			pass

		if len(x)>0:
			return x
		else:
			raise AttributeError

	def loadFile(self,filename,max_num_lines=-1):
		numLines=self._filelines(filename)
		self.head=np.genfromtxt(filename,skip_header=1,skip_footer=numLines-4,names=True)
		skip_lines=0
		if max_num_lines > 0 and max_num_lines<numLines:
			skip_lines=numLines-max_num_lines
		self.data=np.genfromtxt(filename,skip_header=5,names=True,skip_footer=skip_lines)
		self.head_names=self.head.dtype.names
		self.data_names=self.data.dtype.names
		self._loaded=True

	def _filelines(self,filename):
		"""Get the number of lines in a file."""
		f = open(filename, "r+")
		buf = mmap.mmap(f.fileno(), 0)
		lines = 0
		readline = buf.readline
		while readline():
			lines += 1
		f.close()
		return lines


class MESA(object):
	def __init__(self):
		self.hist=data()
		self.prof=data()
		self.prof_ind=""
		self.log_fold=""
	
	def loadHistory(self,f="",filename_in=None,max_model=-1,max_num_lines=-1):
		"""
		Reads a MESA history file.
		
		Optional:
		f: Folder in which history.data exists, if not present uses self.log_fold, if thats
		not set trys the folder LOGS/
		filename_in: Reads the file given by name
		max_model: Maximum model to read into, may help when having to clean files with many retres, backups and restarts by not proccesing data beyond max_model
		
		Returns:
		self.hist.head: The header data in the history file as a structured dtype
		self.hist.data:  The data in the main body of the histor file as a structured dtype
		self.hist.head_names: List of names of the header fields
		self.hist.data_names: List of names of the data fields
		
		Note it will clean the file up of bakups,retries and restarts, prefering to use
		the newest data line.
		"""
		if len(f)==0:
			if len(self.log_fold)==0:
				self.log_fold='LOGS/'
			f=self.log_fold
		else:
			self.log_fold=f+"/"

		if filename_in is None:               
			filename=os.path.join(self.log_fold,'history.data')
		else:
			filename=filename_in

		self.hist.loadFile(filename,max_num_lines)
		
		if max_model>0:
			self.hist.data=self.hist.data[self.hist.model_number<=max_model]

		# Reverse model numbers, we want the unique elements
		# but keeping the last not the first.
		
		#Fix case where we have at end of file numbers:
		# 1 2 3 4 5 3, without this we get the extra 4 and 5
		self.hist.data=self.hist.data[self.hist.model_number<=self.hist.model_number[-1]]
		
		mod_rev=self.hist.model_number[::-1]
		mod_uniq,mod_ind=np.unique(mod_rev,return_index=True)
		self.hist.data=self.hist.data[np.size(self.hist.model_number)-mod_ind-1]

		
	def scrubHistory(self,f="",fileOut="LOGS/history.data.scrubbed"):
		self.loadHistory(f)
		with open(fileOut,'w') as f:
			print(' '.join([str(i) for i in range(1,np.size(self.hist.head_names)+1)]),file=f)
			print(' '.join([str(i) for i in self.hist.head_names]),file=f)
			print(' '.join([str(self.hist.head[i]) for i in self.hist.head_names]),file=f)
			print(" ",file=f)
			print(' '.join([str(i) for i in range(1,np.size(self.hist.data_names)+1)]),file=f)
			print(' '.join([str(i) for i in self.hist.data_names]),file=f)
			for j in range(np.size(self.hist.data)):
				print(' '.join([str(self.hist.data[i][j]) for i in self.hist.data_names]),file=f)	
	
		
	def loadProfile(self,f='',num=None,prof=-1,mode='nearest',silent=False):
		if num==None and prof==-1:
			self._readProfile(f) #f is a filename
			return
		
		if len(f)==0:
			if len(self.log_fold)==0:
				self.log_fold='LOGS/'
			f=self.log_fold
		else:
			self.log_fold=f
		
		if prof>0:
			filename=f+"/profile"+str(int(prof))+".data"
			self._readProfile(filename)
			return
		else:
			self._loadProfileIndex(f) #Assume f is a folder
			if np.count_nonzero(self.prof_ind)==1:
				filename=f+"/profile"+str(int(np.atleast_1d(self.prof_ind["profile"])[0]))+".data"
			else:
				if num==0:
				#Load first model
					filename=f+"/profile"+str(int(self.prof_ind["profile"][0]))+".data"
				elif num<0:
					filename=f+"/profile"+str(int(self.prof_ind["profile"][num]))+".data"
				else:
				#Find profile with mode 'nearest','upper','lower','first','last'
					pos = bisect.bisect_left(self.prof_ind["model"], num)
					if pos == 0 or mode=='first':
						filename=f+"/profile"+str(int(self.prof_ind["profile"][0]))+".data"
					elif pos == np.size(self.prof_ind["profile"]) or mode=='last':
						filename=f+"/profile"+str(int(self.prof_ind["profile"][-1]))+".data"
					elif mode=='lower':
						filename=f+"/profile"+str(int(self.prof_ind["profile"][pos-1]))+".data"
					elif mode=='upper':
						filename=f+"/profile"+str(int(self.prof_ind["profile"][pos]))+".data"
					elif mode=='nearest':
						if self.prof_ind["model"][pos]-num < num-self.prof_ind["model"][pos-1]:
							filename=f+"/profile"+str(int(self.prof_ind["profile"][pos]))+".data"
						else:
							filename=f+"/profile"+str(int(self.prof_ind["profile"][pos-1]))+".data"
					else:
						raise(ValueError,"Invalid mode")
			if not silent:
				print(filename)
			self._readProfile(filename)
			return
			
	#def loadMod(self,filename=None):
		#"""
		#Fails to read a MESA .mod file.
		#"""
		#from io import BytesIO
		
		#count=0
		#with open(filename,'r') as f:
			#for l in f:
				#count=count+1
				#if '!' not in l:
				#break
			#self.mod_head=[]
			#self.mod_head_names=[]
			#self.mod_head.append(int(l.split()[0]))
			#self.mod_head_names.append('mod_version')
			##Blank line
			#f.readline()
			#count=count+1
			##Gap between header and main data
			#for l in f:
				#count=count+1
				#if l=='\n':
				#break
				#self.mod_head.append(l.split()[1])
				#self.mod_head_names.append(l.split()[0])
			#self.mod_dat_names=[]
			#l=f.readline()
			#count=count+1
			#self.mod_dat_names.append('zone')
			#self.mod_dat_names.extend(l.split())
			##Make a dictionary of converters 
			
		#d = {k:self._fds2f for k in range(len(self.mod_dat_names))}	
			
		#self.mod_dat=np.genfromtxt(filename,skip_header=count,
						#names=self.mod_dat_names,skip_footer=5,dtype=None,converters=d)
		
	def iterateProfiles(self,f="",priority=None,rng=[-1.0,-1.0],step=1):
		if len(f)==0:
			if len(self.log_fold)==0:
				self.log_fold='LOGS/'
			f=self.log_fold
		else:
			self.log_fold=f
		#Load profiles index file
		self._loadProfileIndex(f)
		for x in self.prof_ind:
			if priority != None:
				if type(priority) is not list: priority= [ priority ]
				if x["priority"] in priority or 0 in priority:
					self.loadProfile(f=f+"/profile"+str(int(x["profile"]))+".data")
				yield
			if len(rng)==2 and rng[0]>0:
				if x["model"] >=rng[0] and x["model"] <= rng[1] and np.remainder(x["model"]-rng[0],step)==0:
					self.loadProfile(f=f+"/profile"+str(int(x["profile"]))+".data")
				yield
			elif len(rng)>2 and rng[0]>0:
				if x["model"] in rng:
					self.loadProfile(f=f+"/profile"+str(int(x["profile"]))+".data")
				yield
			else:
				self.loadProfile(f=f+"/profile"+str(int(x["profile"]))+".data")
				yield 
				
	def _loadProfileIndex(self,f):
		self.prof_ind=np.genfromtxt(f+"/profiles.index",skip_header=1,names=["model","priority","profile"])

	def _readProfile(self,filename):
		"""
		Reads a MESA profile file.
		
		Required:
		filename: Path to profile to read
		
		Returns:
		self.prof.head: The header data in the profile as a structured dtype
		self.prof.data:  The data in the main body of the profile file as a structured dtype
		self.prof.head_names: List of names of the header fields
		self.prof.data_names: List of names of the data fields
		"""
		self.prof.loadFile(filename)
	
	#def _fds2f(self,x):
		#if isinstance(x, str):
			#f=np.float(x.replace('D','E'))
		#else:
			#f=np.float(x.decode().replace('D','E'))
		#return f
		
	def abun(self,element):
		xx=0
		for ii in range(0,1000):
			try:
				xx=xx+np.sum(self.prof.data[element+str(ii)]*10**m.prof.logdq)
			except:
				pass
		return xx

			
class plot(object):
	def __init__(self):
		self.colors={'clr_Black':[ 0.0, 0.0, 0.0],
					'clr_Blue':[  0.0, 0.0, 1.0],
					'clr_BrightBlue':[  0.0, 0.4, 1.0],
					'clr_LightSkyBlue':[  0.53, 0.808, 0.98],
					'clr_LightSkyGreen':[  0.125, 0.698, 0.668],
					'clr_MediumSpringGreen':[  0.0, 0.98, 0.604],
					'clr_Goldenrod':[  0.855, 0.648, 0.125],
					'clr_Lilac':[  0.8, 0.6, 1.0],
					'clr_Coral':[  1.0, 0.498, 0.312],
					'clr_FireBrick':[  0.698, 0.132, 0.132],
					'clr_RoyalPurple':[  0.4, 0.0, 0.6],
					'clr_Gold':[  1.0, 0.844, 0.0],
					'clr_Crimson':[  0.8, 0.0, 0.2 ],
					'clr_SlateGray':[  0.44, 0.5, 0.565],
					'clr_SeaGreen':[  0.18, 0.545, 0.34],
					'clr_Teal':[  0.0, 0.5, 0.5],
					'clr_LightSteelBlue':[  0.69, 0.77, 0.87],
					'clr_MediumSlateBlue':[  0.484, 0.408, 0.932],
					'clr_MediumBlue':[  0.0, 0.0, 0.804],
					'clr_RoyalBlue':[  0.255, 0.41, 0.884],
					'clr_LightGray':[  0.828, 0.828, 0.828],
					'clr_Silver':[  0.752, 0.752, 0.752],
					'clr_DarkGray':[  0.664, 0.664, 0.664],
					'clr_Gray':[  0.5, 0.5, 0.5],
					'clr_IndianRed':[  0.804, 0.36, 0.36],
					'clr_Tan':[  0.824, 0.705, 0.55],
					'clr_LightOliveGreen':[  0.6, 0.8, 0.6],
					'clr_CadetBlue':[  0.372, 0.62, 0.628],
					'clr_Beige':[  0.96, 0.96, 0.864]}
		
		self.mix_names=['None','Conv','Soften','Over','Semi','Thermo','Rot','Mini','Anon']
		self.mix_col=[self.colors['clr_SeaGreen'], #None
					  self.colors['clr_LightSkyBlue'], #Convection
					  self.colors['clr_LightSteelBlue'], #Softened convection
					  self.colors['clr_SlateGray'], # Overshoot
					  self.colors['clr_Lilac'], #Semi convection
					  self.colors['clr_LightSkyGreen'], #Thermohaline
					  self.colors['clr_BrightBlue'], #Rotation
					  self.colors['clr_Beige'], #Minimum
					  self.colors['clr_Tan'] #Anonymous
					  ]
		
		#Conviently the index of this list is the proton number
		self.elementsPretty=['neut','H', 'He', 'Li', 'Be', 'B', 'C', 'N', 
							'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 
							'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 
							'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 
							'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 
							'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 
							'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 
							'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu',
							'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 
							'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 
							'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 
							'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 
							'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 
							'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 
							'Uub', 'Uut', 'Uuq', 'Uup', 'Uuh', 'Uus', 'Uuo']
		self.elements=[x.lower() for x in self.elementsPretty]
					
		self._getMESAPath()
		
		self.msun=1.9892*10**33
		
		#..names of the stable isotopes
		self.stable_isos = [
	      	'h1','h2','he3','he4','li6','li7','be9','b10',
			'b11','c12','c13','n14','n15','o16','o17','o18',
			'f19','ne20','ne21','ne22','na23','mg24','mg25','mg26',
			'al27','si28','si29','si30','p31','s32','s33','s34',
			's36','cl35','cl37','ar36','ar38','ar40','k39','k40',
			'k41','ca40','ca42','ca43','ca44','ca46','ca48','sc45',
			'ti46','ti47','ti48','ti49','ti50','v50','v51','cr50',
			'cr52','cr53','cr54','mn55','fe54','fe56','fe57','fe58',
			'co59','ni58','ni60','ni61','ni62','ni64','cu63','cu65',
			'zn64','zn66','zn67','zn68','zn70','ga69','ga71','ge70',
			'ge72','ge73','ge74','ge76','as75','se74','se76','se77',
			'se78','se80','se82','br79','br81','kr78','kr80','kr82',
			'kr83','kr84','kr86','rb85','rb87','sr84','sr86','sr87',
			'sr88','y89','zr90','zr91','zr92','zr94','zr96','nb93',
			'mo92','mo94','mo95','mo96','mo97','mo98','mo100','ru96',
			'ru98','ru99','ru100','ru101','ru102','ru104','rh103','pd102',
			'pd104','pd105','pd106','pd108','pd110','ag107','ag109','cd106',
			'cd108','cd110','cd111','cd112','cd113','cd114','cd116','in113',
			'in115','sn112','sn114','sn115','sn116','sn117','sn118','sn119',
			'sn120','sn122','sn124','sb121','sb123','te120','te122','te123',
			'te124','te125','te126','te128','te130','i127','xe124','xe126',
			'xe128','xe129','xe130','xe131','xe132','xe134','xe136','cs133',
			'ba130','ba132','ba134','ba135','ba136','ba137','ba138','la138',
			'la139','ce136','ce138','ce140','ce142','pr141','nd142','nd143',
			'nd144','nd145','nd146','nd148','nd150','sm144','sm147','sm148',
			'sm149','sm150','sm152','sm154','eu151','eu153','gd152','gd154',
			'gd155','gd156','gd157','gd158','gd160','tb159','dy156','dy158',
			'dy160','dy161','dy162','dy163','dy164','ho165','er162','er164',
			'er166','er167','er168','er170','tm169','yb168','yb170','yb171',
			'yb172','yb173','yb174','yb176','lu175','lu176','hf174','hf176',
			'hf177','hf178','hf179','hf180','ta180','ta181','w180','w182',
			'w183','w184','w186','re185','re187','os184','os186','os187',
			'os188','os189','os190','os192','ir191','ir193','pt190','pt192',
			'pt194','pt195','pt196','pt198','au197','hg196','hg198','hg199',
			'hg200','hg201','hg202','hg204','tl203','tl205','pb204','pb206',
			'pb207','pb208','bi209','th232','u235','u238']

		self.solar_is_set=False

		#..anders & grevesse 1989 solar mass fractions
		self._sol_comp_ag89 =[
			7.0573E-01, 4.8010E-05, 2.9291E-05, 2.7521E-01, 6.4957E-10, 
			9.3490E-09, 1.6619E-10, 1.0674E-09, 4.7301E-09, 3.0324E-03, 
			3.6501E-05, 1.1049E-03, 4.3634E-06, 9.5918E-03, 3.8873E-06, 
			2.1673E-05, 4.0515E-07, 1.6189E-03, 4.1274E-06, 1.3022E-04, 
			3.3394E-05, 5.1480E-04, 6.7664E-05, 7.7605E-05, 5.8052E-05, 
			6.5301E-04, 3.4257E-05, 2.3524E-05, 8.1551E-06, 3.9581E-04, 
			3.2221E-06, 1.8663E-05, 9.3793E-08, 2.5320E-06, 8.5449E-07, 
			7.7402E-05, 1.5379E-05, 2.6307E-08, 3.4725E-06, 4.4519E-10, 
			2.6342E-07, 5.9898E-05, 4.1964E-07, 8.9734E-07, 1.4135E-06,
			2.7926E-09, 1.3841E-07, 3.8929E-08, 2.2340E-07, 2.0805E-07, 
			2.1491E-06, 1.6361E-07, 1.6442E-07, 9.2579E-10, 3.7669E-07, 
			7.4240E-07, 1.4863E-05, 1.7160E-06, 4.3573E-07, 1.3286E-05, 
			7.1301E-05, 1.1686E-03, 2.8548E-05, 3.6971E-06, 3.3579E-06, 
			4.9441E-05, 1.9578E-05, 8.5944E-07, 2.7759E-06, 7.2687E-07, 
			5.7528E-07, 2.6471E-07, 9.9237E-07, 5.8765E-07, 8.7619E-08, 
			4.0593E-07, 1.3811E-08, 3.9619E-08, 2.7119E-08, 4.3204E-08, 
			5.9372E-08, 1.7136E-08, 8.1237E-08, 1.7840E-08, 1.2445E-08, 
			1.0295E-09, 1.0766E-08, 9.1542E-09, 2.9003E-08, 6.2529E-08, 
			1.1823E-08, 1.1950E-08, 1.2006E-08, 3.0187E-10, 2.0216E-09, 
			1.0682E-08, 1.0833E-08, 5.4607E-08, 1.7055E-08, 1.1008E-08, 
			4.3353E-09, 2.8047E-10, 5.0468E-09, 3.6091E-09, 4.3183E-08, 
			1.0446E-08, 1.3363E-08, 2.9463E-09, 4.5612E-09, 4.7079E-09, 
			7.7706E-10, 1.6420E-09, 8.7966E-10, 5.6114E-10, 9.7562E-10, 
			1.0320E-09, 5.9868E-10, 1.5245E-09, 6.2225E-10, 2.5012E-10, 
			8.6761E-11, 5.9099E-10, 5.9190E-10, 8.0731E-10, 1.5171E-09, 
			9.1547E-10, 8.9625E-10, 3.6637E-11, 4.0775E-10, 8.2335E-10, 
			1.0189E-09, 1.0053E-09, 4.5354E-10, 6.8205E-10, 6.4517E-10, 
			5.3893E-11, 3.9065E-11, 5.5927E-10, 5.7839E-10, 1.0992E-09, 
			5.6309E-10, 1.3351E-09, 3.5504E-10, 2.2581E-11, 5.1197E-10, 
			1.0539E-10, 7.1802E-11, 3.9852E-11, 1.6285E-09, 8.6713E-10, 
			2.7609E-09, 9.8731E-10, 3.7639E-09, 5.4622E-10, 6.9318E-10, 
			5.4174E-10, 4.1069E-10, 1.3052E-11, 3.8266E-10, 1.3316E-10, 
			7.1827E-10, 1.0814E-09, 3.1553E-09, 4.9538E-09, 5.3600E-09, 
			2.8912E-09, 1.7910E-11, 1.6223E-11, 3.3349E-10, 4.1767E-09, 
			6.7411E-10, 3.3799E-09, 4.1403E-09, 1.5558E-09, 1.2832E-09, 
			1.2515E-09, 1.5652E-11, 1.5125E-11, 3.6946E-10, 1.0108E-09, 
			1.2144E-09, 1.7466E-09, 1.1240E-08, 1.3858E-12, 1.5681E-09, 
			7.4306E-12, 9.9136E-12, 3.5767E-09, 4.5258E-10, 5.9562E-10, 
			8.0817E-10, 3.6533E-10, 7.1757E-10, 2.5198E-10, 5.2441E-10, 
			1.7857E-10, 1.7719E-10, 2.9140E-11, 1.4390E-10, 1.0931E-10, 
			1.3417E-10, 7.2470E-11, 2.6491E-10, 2.2827E-10, 1.7761E-10, 
			1.9660E-10, 2.5376E-12, 2.8008E-11, 1.9133E-10, 2.6675E-10, 
			2.0492E-10, 3.2772E-10, 2.9180E-10, 2.8274E-10, 8.6812E-13, 
			1.4787E-12, 3.7315E-11, 3.0340E-10, 4.1387E-10, 4.0489E-10, 
			4.6047E-10, 3.7104E-10, 1.4342E-12, 1.6759E-11, 3.5397E-10,
			2.4332E-10, 2.8557E-10, 1.6082E-10, 1.6159E-10, 1.3599E-12, 
			3.2509E-11, 1.5312E-10, 2.3624E-10, 1.7504E-10, 3.4682E-10, 
			1.4023E-10, 1.5803E-10, 4.2293E-12, 1.0783E-12, 3.4992E-11, 
			1.2581E-10, 1.8550E-10, 9.3272E-11, 2.4131E-10, 1.1292E-14, 
			9.4772E-11, 7.8768E-13, 1.6113E-10, 8.7950E-11, 1.8989E-10, 
			1.7878E-10, 9.0315E-11, 1.5326E-10, 5.6782E-13, 5.0342E-11, 
			5.1086E-11, 4.2704E-10, 5.2110E-10, 8.5547E-10, 1.3453E-09, 
			1.1933E-09, 2.0211E-09, 8.1702E-13, 5.0994E-11, 2.1641E-09, 
			2.2344E-09, 1.6757E-09, 4.8231E-10, 9.3184E-10, 2.3797E-12,
			1.7079E-10, 2.8843E-10, 3.9764E-10, 2.2828E-10, 5.1607E-10, 
			1.2023E-10, 2.7882E-10, 6.7411E-10, 3.1529E-10, 3.1369E-09, 
			3.4034E-09, 9.6809E-09, 7.6127E-10, 1.9659E-10, 3.8519E-13, 
			5.3760E-11]

#..charge of the stable isotopes

		self._stable_charge  =[
        1,   1,   2,   2,   3,   3,   4,   5,   5,   6,   6,   7,   7, 
         8,   8,   8,   9,  10,  10,  10,  11,  12,  12,  12,  13,  14, 
        14,  14,  15,  16,  16,  16,  16,  17,  17,  18,  18,  18,  19, 
        19,  19,  20,  20,  20,  20,  20,  20,  21,  22,  22,  22,  22, 
        22,  23,  23,  24,  24,  24,  24,  25,  26,  26,  26,  26,  27, 
        28,  28,  28,  28,  28,  29,  29,  30,  30,  30,  30,  30,  31, 
        31,  32,  32,  32,  32,  32,  33,  34,  34,  34,  34,  34,  34, 
        35,  35,  36,  36,  36,  36,  36,  36,  37,  37,  38,  38,  38, 
        38,  39,  40,  40,  40,  40,  40,  41,  42,  42,  42,  42,  42 ,
        42,  42,  44,  44,  44,  44,  44,  44,  44,  45,  46,  46,  46, 
        46,  46,  46,  47,  47,  48,  48,  48,  48,  48,  48,  48,  48, 
        49,  49,  50,  50,  50,  50,  50,  50,  50,  50,  50,  50,  51, 
        51,  52,  52,  52,  52,  52,  52,  52,  52,  53,  54,  54,  54, 
        54,  54,  54,  54,  54,  54,  55,  56,  56,  56,  56,  56,  56, 
        56,  57,  57,  58,  58,  58,  58,  59,  60,  60,  60,  60,  60, 
        60,  60,  62,  62,  62,  62,  62,  62,  62,  63,  63,  64,  64, 
        64,  64,  64,  64,  64,  65,  66,  66,  66,  66,  66,  66,  66, 
        67,  68,  68,  68,  68,  68,  68,  69,  70,  70,  70,  70,  70 ,
        70,  70,  71,  71,  72,  72,  72,  72,  72,  72,  73,  73,  74, 
        74,  74,  74,  74,  75,  75,  76,  76,  76,  76,  76,  76,  76, 
        77,  77,  78,  78,  78,  78,  78,  78,  79,  80,  80,  80,  80, 
        80,  80,  80,  81,  81,  82,  82,  82,  82,  83,  90,  92,  92]


#..number of nucleons (protons and neutrons) in the stable isotopes

		self._stable_a= [ 
         1,   2,   3,   4,   6,   7,   9,  10,  11,  12,  13,  14,  15, 
        16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28, 
        29,  30,  31,  32,  33,  34,  36,  35,  37,  36,  38,  40,  39, 
        40,  41,  40,  42,  43,  44,  46,  48,  45,  46,  47,  48,  49, 
        50,  50,  51,  50,  52,  53,  54,  55,  54,  56,  57,  58,  59, 
        58,  60,  61,  62,  64,  63,  65,  64,  66,  67,  68,  70,  69, 
        71,  70,  72,  73,  74,  76,  75,  74,  76,  77,  78,  80,  82, 
        79,  81,  78,  80,  82,  83,  84,  86,  85,  87,  84,  86,  87, 
        88,  89,  90,  91,  92,  94,  96,  93,  92,  94,  95,  96,  97,
        98, 100,  96,  98,  99, 100, 101, 102, 104, 103, 102, 104, 105, 
       106, 108, 110, 107, 109, 106, 108, 110, 111, 112, 113, 114, 116, 
       113, 115, 112, 114, 115, 116, 117, 118, 119, 120, 122, 124, 121, 
       123, 120, 122, 123, 124, 125, 126, 128, 130, 127, 124, 126, 128, 
       129, 130, 131, 132, 134, 136, 133, 130, 132, 134, 135, 136, 137, 
       138, 138, 139, 136, 138, 140, 142, 141, 142, 143, 144, 145, 146, 
       148, 150, 144, 147, 148, 149, 150, 152, 154, 151, 153, 152, 154, 
       155, 156, 157, 158, 160, 159, 156, 158, 160, 161, 162, 163, 164, 
       165, 162, 164, 166, 167, 168, 170, 169, 168, 170, 171, 172, 173,
       174, 176, 175, 176, 174, 176, 177, 178, 179, 180, 180, 181, 180, 
       182, 183, 184, 186, 185, 187, 184, 186, 187, 188, 189, 190, 192, 
       191, 193, 190, 192, 194, 195, 196, 198, 197, 196, 198, 199, 200, 
       201, 202, 204, 203, 205, 204, 206, 207, 208, 209, 232, 235, 238]


# jcode tells the type progenitors each stable species can have.
# jcode = 0 if the species is the only stable one of that a
#       = 1 if the species can have proton-rich progenitors
#       = 2 if the species can have neutron-rich progenitors
#       = 3 if the species can only be made as itself (eg k40)

		self._jcode = [
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 
         0,   0,   0,   0,   0,   0,   2,   0,   0,   1,   0,   2,   0, 
         3,   0,   1,   0,   0,   0,   2,   2,   0,   1,   0,   1,   0, 
         2,   3,   0,   1,   0,   0,   2,   0,   1,   0,   0,   2,   0, 
         1,   0,   0,   0,   2,   0,   0,   1,   0,   0,   0,   2,   0, 
         0,   1,   0,   0,   2,   2,   0,   1,   1,   0,   2,   2,   2, 
         0,   0,   1,   1,   1,   0,   2,   2,   0,   2,   1,   1,   1, 
         0,   0,   0,   0,   2,   2,   2,   0,   1,   1,   0,   3,   0, 
         2,   2,   1,   1,   0,   1,   0,   2,   2,   0,   1,   1,   0, 
         2,   2,   2,   0,   0,   1,   1,   1,   0,   2,   2,   2,   2, 
         1,   2,   1,   1,   1,   1,   0,   0,   0,   2,   2,   2,   0, 
         2,   1,   1,   1,   3,   0,   2,   2,   2,   0,   1,   1,   1, 
         0,   3,   0,   2,   2,   2,   0,   1,   1,   1,   0,   3,   0, 
         2,   3,   0,   1,   1,   0,   2,   0,   1,   0,   2,   0,   0, 
         2,   2,   1,   0,   1,   0,   1,   2,   2,   0,   0,   1,   1, 
         0,   2,   0,   2,   2,   0,   1,   1,   1,   0,   2,   0,   2, 
         0,   1,   1,   0,   0,   2,   2,   0,   1,   1,   0,   0,   0, 
         2,   2,   0,   3,   1,   1,   0,   0,   0,   2,   3,   0,   1, 
         0,   0,   2,   2,   0,   2,   1,   1,   1,   0,   0,   2,   2, 
         0,   0,   1,   1,   0,   0,   2,   2,   0,   1,   1,   0,   0, 
         0,   0,   2,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0]		
	
	
	def set_solar(self,solar='ag89'):
		if solar=='ag89':
			self.sol_comp=self._sol_comp_ag89
		else:
			raise ValueError("Must pass either ag89")
			
		self.solar_is_set=True
			
	def is_solar_set(self):
		if not self.solar_is_set:
			raise ValueError("Must call set_solar first")
	
	def _getMESAPath(self):
		self.mesa_dir=os.getenv("MESA_DIR")
		#if self.mesa_dir==None:
			#raise ValueError("Must set $MESA_DIR in terminal or call setMESAPath(mesa_dir)")
		
	def setMESAPath(self,mesa_dir):
		self.mesa_dir=mesa_dir
			
	def _loadBurnData(self):
		try:
			dataDir=self.mesa_dir+"/data/star_data/plot_info/"
		except TypeError:
			raise ValueError("Must set $MESA_DIR or call setMESAPath(MESA_DIR)")
		
		self._hburn=np.genfromtxt(dataDir+"hydrogen_burn.data",names=["logRho","logT"])
		self._heburn=np.genfromtxt(dataDir+"helium_burn.data",names=["logRho","logT"])
		self._cburn=np.genfromtxt(dataDir+"carbon_burn.data",names=["logRho","logT"])
		self._oburn=np.genfromtxt(dataDir+"oxygen_burn.data",names=["logRho","logT"])
	
		self._psi4=np.genfromtxt(dataDir+"psi4.data",names=["logRho","logT"])
		self._elect=np.genfromtxt(dataDir+"elect.data",names=["logRho","logT"])
		self._gamma4=np.genfromtxt(dataDir+"gamma_4_thirds.data",names=["logRho","logT"])
		self._kap=np.genfromtxt(dataDir+'kap_rad_cond_eq.data',names=["logRho","logT"])
		self._opal=np.genfromtxt(dataDir+'opal_clip.data',names=["logRho","logT"])
		self._scvh=np.genfromtxt(dataDir+'scvh_clip.data',names=["logRho","logT"])

	def labels(self,label,log=False,center=False):
		l=''
		
		if '$' in label:
			return label
		
		if log or 'log' in label:
			l=r'$\log_{10}\;$'
		if label=='mass':
			l=l+r"$\rm{Mass}\; [M_{\odot}]$"
		if label=='model':
			l=l+r"$\rm{Model\; number}$"
		if 'teff' in label or label=='logT' or '_T' in label:
			if center:
				l=l+r"$T_{eff,c}\; [K]$"
			else:
				l=l+r"$T_{eff}\; [K]$"
		if label=='rho':
			if center:
				l=l+r"$\rho_{c}\; [\rm{g\;cm^{-3}}]$"
			else:
				l=l+r"$\rho\; [\rm{g\;cm^{-3}}]$"
		if label=='log_column_depth':
			l=l+r'$y\; [\rm{g}\; \rm{cm}^{-2}]$'
		if 'lum' in label and not 'column':
			l=l+r'$L\; [L_{\odot}]$'
		if 'star_age' in label:
			l=l+r'T$\;$'
			if 'sec' in label:
				l=l+'[s]'
			if 'hr' in label:
				l=l+'[hr]'
			if 'day' in label:
				l=l+'[day]'
			if 'yr' in label:
				l=l+'[yr]'
		if 'burn' in label:
			l=l+r'$\epsilon_{'+label.split('_')[1].capitalize()+r"}$"
		if label=='pp':
			l=l+r'$\epsilon_{pp}$'
		if label=='tri_alfa':
			l=l+r'$\epsilon_{3\alpha}$'
		if label=='c12_c12':
			l=l+r'$\epsilon_{c12,c12}$'
		if label=='c12_o16':
			l=l+r'$\epsilon_{c12,o16}$'
		if label=='cno':
			l=l+r'$\epsilon_{cno}$'
		if label=='o16_o16':
			l=l+r'$\epsilon_{o16,o16}$'
		if label=='pnhe4':
			l=l+r'$\epsilon_{pnhe4}$'
		if label=='photo':
			l=l+r'$\epsilon_{\gamma}$'
		if label=='other':
			l=l+r'$\epsilon_{other}$'
		if 'abundance' in label:
			l=l+r'$\chi\; [M_{\odot}]$'
				
		if len(l)==0:
			if '$' not in label:
				l=label.replace('_',' ')
			else:
				l=label
			
		return l
		
	def safeLabel(self,label,axis,strip=None):
		outLabel=''
		if label is not None:
			outLabel=label
		else:
			outLabel=self.labels(axis)
		if strip is not None:
			outLabel=outLabel.replace(strip,'')
		return outLabel
		
	def _listAbun(self,data,prefix=''):
		abun_list=[]
		names=data.data_names
		for j in names:
			i=j[len(prefix):]
			if len(i)<=5 and len(i)>=2 and 'burn_' not in j:
				if i[0].isalpha() and (i[1].isalpha() or i[1].isdigit()) and any(char.isdigit() for char in i) and i[-1].isdigit():
					if (len(i)==5 and i[-1].isdigit() and i[-2].isdigit()) or len(i)<5:
						abun_list.append(prefix+i)
			if i=='neut' or i=='prot':
				abun_list.append(prefix+i)
		return abun_list
	
	def _splitIso(self,iso):
		name=''
		mass=''
		for i in iso:
			if i.isdigit():
				mass+=i
			else:
				name+=i
		if name=='neut' or name=='prot':
			mass=1
		return name,int(mass)
	
	def _getIso(self,iso):
		name,mass=self._splitIso(iso)
		if name=='prot':
			p=1
			n=0
		else:
			p=self.elements.index(name)
			n=mass-p
		return name,p,n


	def _listBurn(self,data):
		burnList=[]
		extraBurn=["pp","cno","tri_alfa","c12_c12","c12_O16","o16_o16","pnhe4","photo","other"]
		for i in data.data_names:
			if "burn_" in i or i in extraBurn:
				burnList.append(i)
		return burnList
		
	def _listMix(self,data):
		mixList=["log_D_conv","log_D_semi","log_D_ovr","log_D_th","log_D_thrm","log_D_minimum","log_D_anon","log_D_rayleigh_taylor","log_D_soft"]
		mixListOut=[]		
		for i in data.data_names:
			if i in mixList:
				mixListOut.append(i)
		return mixListOut
	
	def _abunSum(self,m,iso,mass_min=0.0,mass_max=9999.0):
		ind=(m.prof.mass>=mass_min)&(m.prof.mass<=mass_max)
		return np.sum(m.prof.data[iso][ind]*10**m.prof.logdq[ind])*m.prof.star_mass/np.minimum(m.prof.star_mass,mass_max-mass_min)

	def _eleSum(self,m,ele,mass_min=0.0,mass_max=9999.0):
		ind=(m.prof.mass>=mass_min)&(m.prof.mass<=mass_max)
		
		la=self._listAbun(m.prof)
		x=0.0
		for i in la:
			if ele == i[0:len(ele)]:
				x=x+np.sum(m.prof.data[i][ind]*10**m.prof.logdq[ind])*m.prof.star_mass/np.minimum(m.prof.star_mass,mass_max-mass_min)
		return x

	def _setMixRegionsCol(self,kip=True,mix=False):		
		cmap = mpl.colors.ListedColormap(self.mix_col)
	
		cmap.set_over((1., 1., 1.))
		cmap.set_under((0., 0., 0.))
		bounds=[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]
		norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
		return cmap,norm
		
	def _setTicks(self,ax):
		ax.yaxis.set_major_locator(MaxNLocator(5))
		ax.xaxis.set_major_locator(MaxNLocator(5))
		ax.yaxis.set_minor_locator(AutoMinorLocator(10))
		ax.xaxis.set_minor_locator(AutoMinorLocator(10))
	
	def _plotBurnRegions(self,m,ax,x,y,show_x,show_line,yrng=None,ind=None):
		# non 0.0, yellow 1, ornage 10**4, red 10**7
		ylim=ax.get_ylim()
		
		if show_x:
			yy=np.zeros(np.size(x))
			if yrng is not None:
				yy[:]=yrng[0]
			else:
				yy[:]=ylim[0]
			size=240

		yy=y
		size=180
			
		if ind is not None:
			netEng=m.prof.data['net_nuclear_energy'][ind]
		else:
			netEng=m.prof.data['net_nuclear_energy']
		
		ind2=(netEng>=1.0)&(netEng<=4.0)	
		ax.scatter(x[ind2],yy[ind2],c='yellow',s=size,linewidths=0,alpha=1.0)
		ind2=(netEng>=4.0)&(netEng<=7.0)
		ax.scatter(x[ind2],yy[ind2],c='orange',s=size,linewidths=0,alpha=1.0)
		ind2=(netEng>=7.0)
		ax.scatter(x[ind2],yy[ind2],c='red',s=size,edgecolor='none',alpha=1.0)
		ax.set_ylim(ylim)
		
		
	def _plotMixRegions(self,m,ax,x,y,show_x,show_line,yrng=None,ind=None):
		
		if ind is not None:
			if np.count_nonzero(ind)==0:
				return
		
		ylim=ax.get_ylim()
		
		if show_x:
			yy=np.zeros(np.size(x))
			if yrng is not None:
				yy[:]=yrng[0]
			else:
				yy[:]=ylim[0]
			size=150
		else:
			yy=y
			size=60
	
		cmap,norm=self._setMixRegionsCol()
		
		isSet=False
		col=np.zeros(np.size(x))
		for mixLabel in ['mixing_type','conv_mixing_type']:
			try:
				col=m.prof.data[mixLabel]
				isSet=True
				break
			except:
				pass
	
		
		if isSet is None:
			raise(ValueError,"Need mixing type in profile file for showing mix regions, either its mixing_type or conv_mixing_type")
		
		if ind is not None:
			col=col[ind]
			x=x[ind]
			yy=yy[ind]
		
		ax.scatter(x,yy,c=col,s=size,cmap=cmap,norm=norm,linewidths=0)
	
		ax.set_ylim(ylim)
	
	def _annotateLine(self,ax,x,y,num_labels,xmin,xmax,text,line=None,color=None,fontsize=mpl.rcParams['font.size']-12):
		ind=np.argsort(x)
		xx=x[ind]
		yy=y[ind]
		
		ind=(xx>xmin)&(xx<xmax)
		xx=xx[ind]
		yy=yy[ind]
		
		for ii in range(1,num_labels+1):
			if np.size(xx)>1:
				f = interpolate.interp1d(xx,yy)
				xp1=((xmax-xmin)*(ii/(num_labels+1.0)))+xmin
				yp1=f(xp1)
			else:
				xp1=xmin
				yp1=-99*10**9
			if line is None:
				col=color
			else:
				col=line.get_color()
			ax.annotate(text, xy=(xp1,yp1), xytext=(xp1,yp1),color=col,fontsize=fontsize)
	
	def _setYLim(self,ax,yrngIn,yrngOut,rev=False,log=False):
		yrng=[]
		if yrngOut is not None:
			yrng=yrngOut
		else:
			yrng=yrngIn
			
		if rev:
			yrng=yrng[::-1]
		#if (log==True or log=='log') and log!='linear':
			#yrng=np.log10(yrng)
		ax.set_ylim(yrng)
			
	def _setXAxis(self,xx,xmin,xmax,fx):
		x=xx
		if fx is not None:
			x=fx(x)
		
		xrngL=[0,0]
		if xmin is not None:
			xrngL[0]=xmin
		else:
			xrngL[0]=np.min(x)

		if xmax is not None:
			xrngL[1]=xmax
		else:
			xrngL[1]=np.max(x)
			
		ind=(x>=xrngL[0])&(x<=xrngL[1])
			
		return x,xrngL,ind
   
	def _cycleColors(self,ax,colors=None,cmap='',num_plots=0,random_col=False):
		if colors is None:
			c=[cmap(i) for i in np.linspace(0.0,0.9,num_plots)]
		else:
			c=colors
		if random_col:
			random.shuffle(c)
		ax.set_prop_cycle(cycler('color',c))

	def _showBurnData(self,ax):
		self._loadBurnData()
		ax.plot(self._hburn["logRho"],self._hburn["logT"],color=self.colors['clr_Gray'])
		ax.annotate('H burn', xy=(self._hburn["logRho"][-1],self._hburn["logT"][-1]), 
						xytext=(self._hburn["logRho"][-1],self._hburn["logT"][-1]),color=self.colors['clr_Gray'],
						fontsize=mpl.rcParams['font.size']-12)
						
		ax.plot(self._heburn["logRho"],self._heburn["logT"],color=self.colors['clr_Gray'])
		ax.annotate('He burn', xy=(self._heburn["logRho"][-1],self._heburn["logT"][-1]), 
						xytext=(self._heburn["logRho"][-1],self._heburn["logT"][-1]),color=self.colors['clr_Gray'],
						fontsize=mpl.rcParams['font.size']-12)
						
		ax.plot(self._cburn["logRho"],self._cburn["logT"],color=self.colors['clr_Gray'])
		ax.annotate('C burn', xy=(self._cburn["logRho"][-1],self._cburn["logT"][-1]), 
						xytext=(self._cburn["logRho"][-1],self._cburn["logT"][-1]),color=self.colors['clr_Gray'],
						fontsize=mpl.rcParams['font.size']-12)
		
		ax.plot(self._oburn["logRho"],self._oburn["logT"],color=self.colors['clr_Gray'])
		ax.annotate('O burn', xy=(self._oburn["logRho"][-1],self._oburn["logT"][-1]), 
						xytext=(self._oburn["logRho"][-1],self._oburn["logT"][-1]),color=self.colors['clr_Gray'],
						fontsize=mpl.rcParams['font.size']-12)
		
	def _showPgas(self,ax):
		lr1=-8
		lr2=5
		lt1=np.log10(3.2*10**7)+(lr1-np.log10(0.7))/3.0
		lt2=np.log10(3.2*10**7)+(lr2-np.log10(0.7))/3.0
		ax.plot([lr1,lr2],[lt1,lt2],color=self.colors['clr_Gray'])
		ax.annotate(r'$P_{rad}\approx P_{gas}$', xy=(-4.0,6.5), 
						xytext=(-4.0,6.5),color=self.colors['clr_Gray'],
						fontsize=mpl.rcParams['font.size']-12)
						
	def _showDegeneracy(self,ax):
		ax.plot(self._psi4["logRho"],self._psi4["logT"],color=self.colors['clr_Gray'])
		ax.annotate(r'$\epsilon_F/KT\approx 4$', xy=(2.0,6.0), 
						xytext=(2.0,6.0),color=self.colors['clr_Gray'],
						fontsize=mpl.rcParams['font.size']-12)

	def _showGamma4(self,ax):
		ax.plot(self._gamma4["logRho"],self._gamma4["logT"],color=self.colors['clr_Crimson'])
		ax.annotate(r'$\Gamma_{1} <4/3$', xy=(3.8,9.2), 
						xytext=(3.8,9.2),color=self.colors['clr_Crimson'],
						fontsize=mpl.rcParams['font.size']-12)
						
	def _showEOS(self,ax):
		logRho1 =  2.7
		logRho2 =  2.5
		logRho3 =  -1.71
		logRho4  = -2.21
		logRho5  = -9.0
		logRho6  = -9.99
		logRho7  = -12
		logT1  =  7.7
		logT2 =   7.6
		logT3  =  4.65
		logT4  =  4.75
		logT5  =  3.60
		logT6  =  3.50
		logT7 =   2.3
		logT8  =  2.2
		
		ax.plot([logRho2,logRho7], [logT1,logT1],color=self.colors['clr_LightSkyGreen'])         
		ax.plot([logRho2,logRho7], [logT2,logT2],color=self.colors['clr_LightSkyGreen'])         
		ax.plot([logRho2,logRho1], [logT1,logT2],color=self.colors['clr_LightSkyGreen'])         
		ax.plot([logRho1,logRho1], [logT2,logT3],color=self.colors['clr_LightSkyGreen'])         

		ax.plot([logRho4,logRho5], [logT7,logT7],color=self.colors['clr_LightSkyGreen'])         
		ax.plot([logRho4,logRho5], [logT8,logT8],color=self.colors['clr_LightSkyGreen'])         
		ax.plot([logRho4,logRho3], [logT8,logT7],color=self.colors['clr_LightSkyGreen'])         
		ax.plot([logRho6,logRho5], [logT7,logT8],color=self.colors['clr_LightSkyGreen'])       
		
		ax.plot([logRho2,logRho2], [logT2,logT4],color=self.colors['clr_LightSkyGreen'])       
		ax.plot([logRho3,logRho1], [logT7,logT3],color=self.colors['clr_LightSkyGreen'])       
		ax.plot([logRho4,logRho2], [logT7,logT4],color=self.colors['clr_LightSkyGreen'])       

		ax.plot([logRho5,logRho5], [logT7,logT6],color=self.colors['clr_LightSkyGreen'])       
		ax.plot([logRho6,logRho6], [logT7,logT6],color=self.colors['clr_LightSkyGreen'])       
		ax.plot([logRho5,logRho6], [logT6,logT5],color=self.colors['clr_LightSkyGreen'])     
			
		ax.plot([logRho6,logRho7], [logT5,logT5],color=self.colors['clr_LightSkyGreen'])       
		ax.plot([logRho6,logRho7], [logT6,logT6],color=self.colors['clr_LightSkyGreen'])  
		
		
		logRho0 = logRho1
		logRho1 = 2.2
		logRho2 = 1.2
		logRho3 = -2.0
		logRho4 = -3.8
		logRho5 = -5.8
		logRho6 = -6.8
		logRho7 = -10
		logT1 = 6.6
		logT2 = 6.5
		logT3 = 4.0
		logT4 = 3.4
		logT5 = 3.3

		ax.plot([logRho0, logRho2],[logT1, logT1],color=self.colors['clr_LightSkyBlue'])          
		ax.plot([logRho2, logRho4],[logT1, logT3],color=self.colors['clr_LightSkyBlue'])             
		ax.plot([logRho4, logRho5],[logT3, logT4],color=self.colors['clr_LightSkyBlue'])            
		ax.plot([logRho5, logRho7],[logT4, logT4],color=self.colors['clr_LightSkyBlue'])             

		ax.plot([logRho0, logRho1],[logT2, logT2],color=self.colors['clr_LightSkyBlue'])            
		ax.plot([logRho1, logRho3],[logT2, logT3],color=self.colors['clr_LightSkyBlue'])             
		ax.plot([logRho3, logRho5],[logT3, logT5],color=self.colors['clr_LightSkyBlue'])             
		ax.plot([logRho5, logRho7],[logT5, logT5],color=self.colors['clr_LightSkyBlue']) 
		
		ax.annotate('HELM', xy=(8.6,8.6), 
						xytext=(8.6,8.6),color=self.colors['clr_Gray'],
						fontsize=mpl.rcParams['font.size']-12)
		ax.annotate('OPAL', xy=(-7.2, 5.8), 
						xytext=(-7.2, 5.8),color=self.colors['clr_Gray'],
						fontsize=mpl.rcParams['font.size']-12)
		ax.annotate('SCVH', xy=(-0.8, 3.7), 
						xytext=(-0.8, 3.7),color=self.colors['clr_Gray'],
						fontsize=mpl.rcParams['font.size']-12)
		ax.annotate('PC', xy=(7.1, 5.1), 
						xytext=(7.1, 5.1),color=self.colors['clr_Gray'],
						fontsize=mpl.rcParams['font.size']-12)
						
	def _showBurnMixLegend(self,ax,mix=True,burn=True):
		
		label=[]
		color=[]

		if burn:
			label.append(r'$>1\; \rm{erg}^{-1}\;s^{-1}$')
			color.append(self.colors['clr_Gold'])
			label.append(r'$>1000\; \rm{erg}^{-1}\;s^{-1}$')
			color.append(self.colors['clr_Coral'])
			label.append(r'$>10^7\; \rm{erg}^{-1}\;s^{-1}$')
			color.append(self.colors['clr_Crimson'])

		if mix:
			cmap,norm=self._setMixRegionsCol(mix=True)
			for i,j in zip(self.mix_names,self.mix_col):
				label.append(i)
				color.append(j)
				
		xlim=ax.get_xlim()
		ylim=ax.get_ylim()
		for i,j, in zip(label,color):
			ax.plot([0,0],[0,0],color='w',label=i,alpha=0.0)
				
		leg=ax.legend(framealpha = 0,labelspacing=0.0,numpoints=1,loc=4,handlelength=1)
		for text,i,j in zip(leg.get_texts(),label,color):
			plt.setp(text, color = j,fontsize=16)
			text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                       path_effects.Normal()])
	
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)
	
	def _plotCoreLoc(self,prof,ax,xaxis,x,ymin,ymax,linecol='k',coreMasses=None):
		if coreMasses is None:
			coreMasses=['he_core_mass','c_core_mass','o_core_mass','si_core_mass','fe_core_mass']
		
		for cm in coreMasses:
			#Find cell where we have core mass and use that to index the actual x axis
			pos = bisect.bisect_right(prof.data["mass"][::-1], prof.head[cm])
			if pos < np.size(prof.data[xaxis]) and pos > 0 and prof.head[cm] >0.0:
				pos=np.size(prof.data['mass'])-pos
				ax.plot([prof.data[xaxis][pos],prof.data[xaxis][pos]],[ymin,ymax],'--',color=linecol)
				xp1=prof.data[xaxis][pos]
				yp1=0.95*(ymax-ymin)+ymin
				ax.annotate(cm.split('_')[0], xy=(xp1,yp1), xytext=(xp1,yp1),color=linecol,fontsize=mpl.rcParams['font.size']-12)
				
	def _showMassLoc(self,m,fig,ax,x,modInd):
		coreMasses=['he_core_mass','c_core_mass','o_core_mass','si_core_mass','fe_core_mass']
		labels=['He','C','O','Si','Fe']
		col=['clr_Teal','clr_LightOliveGreen','clr_SeaGreen','clr_Lilac','clr_Crimson']
		
		for i,j in zip(coreMasses,col):
			y=m.hist.data[i][modInd]
			if np.any(y):
				ax.plot(x,y,color=self.colors[j],linewidth=5)
				
		self._addExtraLabelsToAxis(fig,labels,[self.colors[i] for i in col],num_left=0,num_right=len(labels),right_pad=50)
		
	def _showMassLocHist(self,m,fig,ax,x,y,modInd):
		coreMasses=['he_core_mass','c_core_mass','o_core_mass','si_core_mass','fe_core_mass']
		labels=['He','C','O','Si','Fe']
		col=['clr_Teal','clr_LightOliveGreen','clr_SeaGreen','clr_Lilac','clr_Crimson']
		
		out=[]
		outc=[]
		for i,j,l in zip(coreMasses,col,labels):
			ind=m.hist.data[i][modInd]>0.0
			if np.count_nonzero(ind):
				ax.plot([x[ind][0],x[ind][0]],ax.get_ylim(),'--',color=self.colors[j],linewidth=2)
				out.append(x[ind][0])
				outc.append(l)
		
		ax2=ax.twiny()
		ax2.plot(ax.get_xlim(),ax.get_ylim())
		ax2.cla()
		ax2.set_xlim(ax.get_xlim())
		ax2.set_xticks(out)
		ax2.set_xticklabels(outc)
		plt.sca(ax)
		
	def _findShockLoc(self,prof,ind):
		cs=prof.data['csound'][ind]
		vel=prof.data['velocity'][ind]
		#Find location of shock
		s=np.count_nonzero(cs)
		fs=False
		k=-1
		for k in range(0,s-1):
			if vel[k+1]>=cs[k] and vel[k]<cs[k]:
				fs=True
				break
		
		if not fs:
			for k in range(s-1,0):
				if vel[k+1]>=-cs[k] and vel[k]<-cs[k]:
					fs=True
					break
		return fs,k		
		
		
	def _showShockLoc(self,prof,fig,ax,xaxis,yrng,ind):
		fs,k=self._findShockLoc(prof,ind)
		#check we are either side of shock
		if fs:
			xx=[xaxis[ind][k],xaxis[ind][k]]
			ax.plot(xx,yrng,'--',color=self.colors['clr_DarkGray'],linewidth=2)
			
	def _getMassFrac(self,m,i,massInd):
		return np.sum(m.prof.data[i][massInd]*10**(m.prof.logdq[massInd]))
		
	def _getMassIso(self,m,i,massInd):
		mcen=0.0
		try:
			mcen=m.prof.M_center
		except AttributeError:
			pass
		mass=m.prof.star_mass-mcen/self.msun	
			
		return self._getMassFrac(m,i,massInd)*mass
		
	def _addExtraLabelsToAxis(self,fig,labels,colors=None,num_left=0,num_right=0,left_pad=85,right_pad=85):

		total_num=len(labels)
		
		if colors is None:
			colors=['k']
			colors=colors*total_num
		
		scale=2.5
		if num_left > 0:
			for i in range(num_left):
				axis=fig.add_subplot(num_left,3,(i*3)+1)
				axis.spines['top'].set_visible(False)
				axis.spines['right'].set_visible(False)
				axis.spines['bottom'].set_visible(False)
				axis.spines['left'].set_visible(False)
				axis.yaxis.set_major_locator(plt.NullLocator())
				axis.xaxis.set_major_locator(plt.NullLocator())
				axis.yaxis.set_minor_locator(plt.NullLocator())
				axis.xaxis.set_minor_locator(plt.NullLocator())
				axis.patch.set_facecolor('None')
				axis.plot(0,0,color='w')
				scale=2.0
				text=axis.set_ylabel(labels[i],color=colors[i], labelpad=left_pad,fontsize=16)
				text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),
                       path_effects.Normal()])
			
		flip=False
		if num_right > 0:
			for j in range(num_right):
				i=num_left+j
				axis=fig.add_subplot(num_right,3,((j+1)*3))
				axis.spines['top'].set_visible(False)
				axis.spines['right'].set_visible(False)
				axis.spines['bottom'].set_visible(False)
				axis.spines['left'].set_visible(False)
				axis.yaxis.set_major_locator(plt.NullLocator())
				axis.xaxis.set_major_locator(plt.NullLocator())
				axis.yaxis.set_minor_locator(plt.NullLocator())
				axis.xaxis.set_minor_locator(plt.NullLocator())
				axis.patch.set_facecolor('None')
				axis.plot(0,0,color='w')
				axis2=axis.twinx()
				axis2.spines['top'].set_visible(False)
				axis2.spines['right'].set_visible(False)
				axis2.spines['bottom'].set_visible(False)
				axis2.spines['left'].set_visible(False)
				axis2.yaxis.set_major_locator(plt.NullLocator())
				axis2.xaxis.set_major_locator(plt.NullLocator())
				axis2.yaxis.set_minor_locator(plt.NullLocator())
				axis2.xaxis.set_minor_locator(plt.NullLocator())
				axis2.patch.set_facecolor('None')
				text=axis2.set_ylabel(labels[i],color=colors[i], labelpad=right_pad,fontsize=16)
				text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),
                       path_effects.Normal()])
			
		
	def _addMixLabelsAxis(self,fig):
		self._setMixRegionsCol(kip=True)
		self._addExtraLabelsToAxis(fig,labels=self.mix_names,colors=self.mix_col,num_left=len(self.mix_names))
		
	def setTitle(self,ax,show_title_name=False,show_title_model=False,show_title_age=False,
				name=None,model=None,age=None,age_units=None,
				fontCent=mpl.rcParams['font.size']-6,fontOther=mpl.rcParams['font.size']-12):
		if show_title_name:
			ax.set_title(name,loc="center",fontsize=fontCent)
		if show_title_model:
			ax.set_title("Model "+str(np.int(model)),loc="right",fontsize=fontOther)
		if show_title_age:
			s="age "+"{:8.4e}".format(np.float(age))
			if age_units is None:
				s=s+" yrs"
			else:
				s=s+" "+age_units
			ax.set_title(s,loc="left",fontsize=fontOther)
			

	def _plotAnnotatedLine(self,ax,x,y,fy,xmin,xmax,ymin=None,ymax=None,annotate_line=False,label=None,
							points=False,xlog=False,ylog=False,xrev=False,yrev=False,linecol=None,
							linewidth=2,num_labels=5,linestyle='-'):
			if xlog:
				x=np.log10(x)
			if ylog:
				y=np.log10(y)
			if fy is not None:
				y=fy(y)
			
			if ymin is None or ymax is None:
				ymin=np.nanmin(y)
				ymax=np.nanmax(y)
			if xmin is None or xmax is None:
				xmin=np.nanmin(x)
				xmax=np.nanmax(x)
			
			y[np.logical_not(np.isfinite(y))]=ymin-(ymax-ymin)
			if linecol is None:
				line, =ax.plot(x,y,linestyle=linestyle,linewidth=linewidth)
			else:
				line, =ax.plot(x,y,linestyle=linestyle,color=linecol,linewidth=linewidth)
			if points:
				ax.scatter(x,y)
			if annotate_line:
				self._annotateLine(ax,x,y,num_labels,xmin,xmax,label,line)
			
			if xrev:
				ax.set_xlim(xmax,xmin)
			else:
				ax.set_xlim(xmin,xmax)
			if yrev:
				ax.set_ylim(ymax,ymin)
			else:
				ax.set_ylim(ymin,ymax)
			self._setTicks(ax)
	
			return x,y
		
	def _setupHist(self,fig,ax,m,minMod,maxMod):
		
		if m.hist._loaded is False:
			raise ValueError("Must call loadHistory first")
		
		if fig==None:
			fig=plt.figure(figsize=(12,12))
		if ax==None:
			ax=fig.add_subplot(111)
		
		if maxMod<0:
			maxMod=m.hist.data["model_number"][-1]
		modelIndex=(m.hist.data["model_number"]>=minMod)&(m.hist.data["model_number"]<=maxMod)		
		
		return fig,ax,modelIndex
		
	def _setupProf(self,fig,ax,m,model,label='plot'):
		
		if m.prof._loaded is False:
			raise ValueError("Must call loadProfile first")
		
		if fig==None:
			fig=plt.figure(figsize=(12,12))
		if ax==None:
			ax=fig.add_subplot(111,label=label)
		#m.loadProfile(num=int(model))
		
		if model is not None:
			try:
				if m.prof.head["model_number"]!=model:
					m.loadProfile(num=int(model))
			except:
				m.loadProfile(num=int(model))
		return fig,ax
	
	def _setYLabel(self,fig,ax,ylabel,default=None,color='k'):
		ax.set_ylabel(self.safeLabel(ylabel,default),color=color)
			
	def _setXLabel(self,fig,ax,xlabel,default=None,color='k'):	
		ax.set_xlabel(self.safeLabel(xlabel,default),color=color)
	
	def _plotY2(self,fig,ax,x,data,xrngL,xlog,xrev,mInd,y2=None,y2rng=[None,None],fy2=None,y2Textcol=None,y2label=None,y2rev=False,y2log=False,y2col='k',points=False):

		if y2 is not None:
			ax2=ax.twinx()
			ax2.set_label('abun_ax2')
			y=data[y2][mInd]
			px,py=self._plotAnnotatedLine(ax2,x[mInd],y,fy2,xrngL[0],xrngL[1],y2rng[0],y2rng[1],
					annotate_line=False,label=self.safeLabel(y2label,y2),
					points=points,xlog=xlog,ylog=y2log,xrev=xrev,
					yrev=y2rev,linecol=y2col)  

			if y2Textcol is None:
				y2labcol=y2col
			else:
				y2labcol=y2Textcol
			
			self._setYLabel(fig,ax2,y2label,y2, color=y2labcol)

		plt.sca(ax)


	def _decay2Stable(self,m,massInd):
		abun_list=self._listAbun(m.prof)
		
		res=[]
		for i,j,p in zip(self.stable_isos,self._stable_a,self._stable_charge):
			res.append({'name':i,'p':p,'a':j,'mass':0})

		msum=0
		for i in abun_list:
			element,p,n=self._getIso(i)
			a=p+n
			massFrac=self._getMassFrac(m,i,massInd)
			for idj,j in enumerate(res):
				if j['a'] != a:
					continue
				if (self._jcode[idj]==0 or 
					(p>=self._stable_charge[idj] and self._jcode[idj]==1) or 
					(p<=self._stable_charge[idj] and self._jcode[idj]==2) or 
					(p==self._stable_charge[idj] and self._jcode[idj]==3)):
					res[idj]['mass']=res[idj]['mass']+massFrac
					msum=msum+massFrac
			
		for i in res:
			i['mass']=i['mass']/msum
			
		return res
		
	def get_solar(self):
		self.is_solar_set()
		res=[]
		for i,j,p,m in zip(self.stable_isos,self._stable_a,self._stable_charge,self.sol_comp):
			res.append({'name':i,'p':p,'a':j,'mass':m})
		return res

	
	def plotAbun(self,m,model=None,show=True,ax=None,xaxis='mass',xmin=None,xmax=None,yrng=[-3.0,1.0],
				cmap=plt.cm.gist_ncar,num_labels=3,xlabel=None,points=False,abun=None,abun_random=False,
				show_burn=False,show_mix=False,fig=None,fx=None,fy=None,modFile=False,
				show_title_name=False,show_title_model=False,show_title_age=False,annotate_line=True,linestyle='-',
				colors=None,ylabel=None,title=None,show_shock=False,
				y2=None,y2rng=[None,None],fy2=None,y2Textcol=None,y2label=None,y2rev=False,y2log=False,y2col='k',xlog=False,xrev=False):
		
		fig,ax=self._setupProf(fig,ax,m,model,label='abun_ax1')
	
			
		if modFile:
			x,xrngL,mInd=self._setXAxis(np.cumsum(m.mod_dat["dq"][::-1])*m._fds2f(m.mod_head[1]),xmin,xmax,fx)
		else:
			x,xrngL,mInd=self._setXAxis(m.prof.data[xaxis],xmin,xmax,fx)

		
		if abun is None:
			abun_list=self._listAbun(m.prof)
			log=''
		else:
			abun_list=abun
			log=""
			
		abun_log=True
		if len(log)>0:
			abun_log=False
			
		num_plots=len(abun_list)
		#Helps when we have many elements not on the plot that stretch the colormap
		if abun_random:
			random.shuffle(abun_list)

		self._cycleColors(ax,colors,cmap,num_plots)
		
		for i in abun_list:
			self._plotAnnotatedLine(ax=ax,x=x,y=m.prof.data[i],fy=fy,xmin=xrngL[0],xmax=xrngL[1],
				ymin=yrng[0],ymax=yrng[1],annotate_line=annotate_line,
				label=self.safeLabel(None,i),points=points,ylog=abun_log,num_labels=num_labels,
				linestyle=linestyle,xrev=xrev,xlog=xlog)
			
		if show_burn:
			self._plotBurnRegions(m,ax,x,m.prof.mass,show_line=False,show_x=True,yrng=yrng,ind=mInd)

		if show_mix:
			self._plotMixRegions(m,ax,x,m.prof.mass,show_line=False,show_x=True,yrng=yrng,ind=mInd)
			
		if show_shock:
			self._showShockLoc(m.prof,fig,ax,x,yrng,mInd)
			
		if y2 is not None:
			self._plotY2(fig,ax,x,m.prof.data,xrngL,xlog,xrev,mInd,y2,y2rng,fy2,y2Textcol,y2label,y2rev,y2log,y2col,points)

			
		self._setXLabel(fig,ax,xlabel,xaxis)
		self._setYLabel(fig,ax,ylabel,r'$\log_{10}$ Abundance')
			
		
		if title is not None:
			ax.set_title(title)
		elif show_title_name or show_title_model or show_title_age:
			self.setTitle(ax,show_title_name,show_title_model,show_title_age,'Abundances',m.prof.head["model_number"],m.prof.head["star_age"])
		
		
		if show:
			plt.show()
			
	def plotAbunByA(self,m,m2=None,model=None,show=True,ax=None,xmin=None,xmax=None,mass_range=None,abun=None,
					fig=None,show_title_name=False,show_title_model=False,show_title_age=False,
					cmap=plt.cm.gist_ncar,colors=None,abun_random=False,
					line_labels=True,yrng=None):
		
		fig,ax=self._setupProf(fig,ax,m,model)
				
		if mass_range is None:
			mass_range=[0.0,m.prof.star_mass]
		
		if abun is None:
			abun_list=self._listAbun(m.prof)
			log=''
		else:
			abun_list=abun
			log=""
			
		ax.set_yscale('log')
			
		abun_log=True
		if len(log)>0:
			abun_log=False
			
		massInd=(m.prof.mass>=mass_range[0])&(m.prof.mass<=mass_range[1])
		
		if m2 is not None:
			massInd2=(m2.prof.mass>=mass_range[0])&(m2.prof.mass<=mass_range[1])
		
		if xmin is None:
			xmin=-1
		if xmax is None:
			xmax=999999
	
			
		data=[]
		ys=[]
		
		for i in abun_list:
			name,mass=self._splitIso(i)
			if name=='neut' or name=='prot':
				continue
			if mass >= xmin and mass <= xmax:
				total_mass=self._getMassFrac(m,i,massInd)
				total_mass2=1.0
				if m2 is not None:
					total_mass2=self._getMassFrac(m2,i,massInd2)
				data.append({'name':name,'mass':mass,
							'totmass1':total_mass,
							'totmass2':total_mass2,
							'rel':total_mass/total_mass2})
				ys.append(total_mass/total_mass2)
		
		uniq_names=set(dic['name'] for dic in data)
		sorted_names=sorted(uniq_names,key=self.elements.index)
		
		self._cycleColors(ax,colors,cmap,len(uniq_names),abun_random)
		
		ymax=np.max(ys)
		ymin=np.min(ys)
		
		if yrng is not None:
			ymax=yrng[1]
			ymin=yrng[0]
		
			
		levels=[-1,0,1]
		for idj,j in enumerate(sorted_names):
			x=[]
			y=[]
			line,=ax.plot(1,1)
			col=line.get_color()
			for i in data:
				if i['name']==j:
					if i['rel']<=ymax and i['rel'] >=ymin:
						ax.scatter(i['mass'],i['rel'],color=col)
						x.append(i['mass'])
						y.append(i['rel'])	
			x=np.array(x)
			y=np.array(y)
			if np.size(x)==0:
				continue
			
			ind=np.argsort(x)
			if np.count_nonzero(ind)>1:
				x=x[ind]
				y=y[ind]	
			line,=ax.plot(x,y,linewidth=2,color=col)
			switch=levels[np.mod(idj,3)]
			ax.text(np.median(x),ymax+(switch*0.25*(ymax-ymin)),j,color=col)
				
		ax.set_xlabel("A")
		if m2 is None:
			ax.set_ylabel(r'$\log_{10}$ Abundance')
		else:
			ax.set_ylabel(r'$\log_{10}\left(\frac{\rm{Abun}_1}{\rm{Abun}_2}\right)$')
		
		if show_title_name or show_title_model or show_title_age:
			self.setTitle(ax,show_title_name,show_title_model,show_title_age,'Production',m.prof.head["model_number"],m.prof.head["star_age"])
		
		ax.set_xlim(0,ax.get_xlim()[1])
	
		if yrng is not None:
			ax.set_ylim(ymin,ymax+0.5*(ymax-ymin))
		else:
			ax.set_ylim(ymin,ymax)
	
		if show:
			plt.show()			


	def plotAbunByA_Stable(self,m,m2=None,model=None,show=True,ax=None,xmin=None,xmax=None,mass_range=None,mass_range2=None,abun=None,
					fig=None,show_title_name=False,show_title_model=False,show_title_age=False,
					cmap=plt.cm.gist_ncar,colors=None,abun_random=False,
					line_labels=True,yrng=None):
		
		fig,ax=self._setupProf(fig,ax,m,model)
				
		if mass_range is None:
			mass_range=[0.0,m.prof.star_mass]
			
		massInd=(m.prof.mass>=mass_range[0])&(m.prof.mass<=mass_range[1])
			
		ax.set_yscale('log')
			
			
		abun=self._decay2Stable(m,massInd)
		
		abun_solar=self.get_solar()
		
		if len(abun_solar) != len(abun):
			raise ValueError("Bad length for solar data "+len(abun_solar)+","+len(abun))
		
		if m2 is not None:
			if mass_range2 is None:
				massInd2=(m2.prof.mass>=mass_range[0])&(m2.prof.mass<=mass_range[1])
			else:
				massInd2=(m2.prof.mass>=mass_range2[0])&(m2.prof.mass<=mass_range2[1])
			abun2=self._decay2Stable(m2,massInd2)
		else:
			abun2=[]
		
		if xmin is None:
			xmin=-1
		if xmax is None:
			xmax=999999
	
			
		data=[]
		ys=[]
		
		for k in range(len(abun)):
			i=abun[k]
			j=abun_solar[k]
			try:
				l=abun2[k]
			except IndexError:
				l=0
			name,mass=self._splitIso(i['name'])
			if mass >= xmin and mass <= xmax:
				total_mass=i['mass']
				total_massSol=j['mass']
				try:
					total_mass2=l['mass']/total_massSol
				except TypeError:
					total_mass2=1.0
					
				if total_mass==0.0 and total_mass2==0.0:
					continue
					
				if total_mass==0.0 or total_mass2==0.0:
					print("Skipping "+i['name']+" mass frac is 0")
					continue
				
				data.append({'name':name,'mass':mass,
							'totmass1':total_mass,
							'totmass2':total_mass2,
							'totmasssol':total_massSol,
							'rel':(total_mass/total_massSol)/total_mass2})
				ys.append(data[-1]['rel'])
		
		uniq_names=set(dic['name'] for dic in data)
		sorted_names=sorted(uniq_names,key=self.elements.index)
		
		self._cycleColors(ax,colors,cmap,len(uniq_names),abun_random)
		
		ymax=np.max(ys)
		ymin=np.min(ys)
		
		if yrng is not None:
			ymax=yrng[1]
			ymin=yrng[0]
		
		levels=[-1,0,1]
		for idj,j in enumerate(sorted_names):
			x=[]
			y=[]
			line,=ax.plot(1,1)
			col=line.get_color()
			for i in data:
				if i['name']==j:
					if i['rel']<=ymax and i['rel'] >=ymin:
						ax.scatter(i['mass'],i['rel'],color=col)
						x.append(i['mass'])
						y.append(i['rel'])	
			x=np.array(x)
			y=np.array(y)
			if np.size(x)==0:
				continue
			
			ind=np.argsort(x)
			if np.count_nonzero(ind)>1:
				x=x[ind]
				y=y[ind]	
			line,=ax.plot(x,y,linewidth=2,color=col)
			switch=levels[np.mod(idj,3)]
			ax.text(np.median(x),ymax+(switch*0.25*(ymax-ymin)),j,color=col)
				
		ax.set_xlabel("A")
		if m2 is None:
			ax.set_ylabel(r'$\log_{10}\left(\frac{\rm{Abun}}{\rm{Abun}_{Sol}}\right)$')
		else:
			ax.set_ylabel(r'$\log_{10}\left(\frac{\rm{m_1}}{\rm{m}_{2}}\right)$')
		
		if show_title_name or show_title_model or show_title_age:
			self.setTitle(ax,show_title_name,show_title_model,show_title_age,'Production',m.prof.head["model_number"],m.prof.head["star_age"])
		
		ax.set_xlim(0,ax.get_xlim()[1])
	
		if yrng is not None:
			ax.set_ylim(ymin,ymax+0.5*(ymax-ymin))
		else:
			ax.set_ylim(ymin,ymax)
	
		if show:
			plt.show()			

			
	def plotAbunPAndN(self,m,model=None,show=True,ax=None,xmin=None,xmax=None,mass_range=None,abun=None,
					num_labels=3,fig=None,show_title_name=False,show_title_model=False,show_title_age=False,
					cmap=plt.cm.jet,colors=None,abun_random=False,abun_scaler=None,line_labels=True,mass_frac_lim=10**-10):
		
		fig,ax=self._setupProf(fig,ax,m,model)
				
		if mass_range is None:
			ymin=0.0
			ymax=m.prof.star_mass
		else:
			ymin=mass_range[0]
			ymax=mass_range[1]
		
		if abun is None:
			abun_list=self._listAbun(m.prof)
			log=''
		else:
			abun_list=abun
			log=""
			
		abun_log=True
		if len(log)>0:
			abun_log=False
			
		massInd=(m.prof.mass>=ymin)&(m.prof.mass<=ymax)
		
		if xmin is None:
			xmin=-1
		if xmax is None:
			xmax=999999
		
		name_all=[]
		name=[]
		proton=[]
		neutron=[]
		for i in abun_list:
			na,pr,ne=self._getIso(i)
			name.append(na)
			proton.append(pr)
			neutron.append(ne)

	
		proton=np.array(proton)
		neutron=np.array(neutron)
		
		outArr=np.zeros((10,10))
		outArr[:]=np.nan
	
		mf=[]	
		for i in abun_list:
			na,pr,ne=self._getIso(i)
			idx=name.index(na)
			massFrac=np.log10(np.sum(m.prof.data[i][massInd]*10**(m.prof.logdq[massInd])))
			mf.append(massFrac)
		
		min_col=np.maximum(np.log10(mass_frac_lim),np.min(mf))
		max_col=np.minimum(np.max(mf),0.0)

		for i in abun_list:
			na,pr,ne=self._getIso(i)
			idx=name.index(na)
			massFrac=np.log10(np.sum(m.prof.data[i][massInd]*10**(m.prof.logdq[massInd])))
			if massFrac >=np.log10(mass_frac_lim):
				ax.add_patch(mpl.patches.Rectangle((float(ne-0.5),float(pr-0.5)),1.0,1.0,facecolor=cmap((massFrac-min_col)/(max_col-min_col))))
			else:
				ax.add_patch(mpl.patches.Rectangle((float(ne-0.5),float(pr-0.5)),1.0,1.0,fill=False))
			#outArr[ne,pr]=massFrac
			
		norm = mpl.colors.Normalize(vmin=min_col, vmax=max_col)
		im=ax.imshow(outArr, aspect='auto',cmap=cmap, norm=norm)
		cb = fig.colorbar(im,ax=ax,cmap=cmap,norm=norm)
		cb.solids.set_edgecolor("face")
		cb.set_label('Log Mass Frac')

		ax.set_xlabel('Neutrons')
		ax.set_ylabel('Protons')
		ax.set_xlim(neutron.min()-1,neutron.max()+1)
		ax.set_ylim(proton.min()-1,proton.max()+1)

		if show:
			plt.show()
			
			
	def plotCenterAbun(self,m,model=None,show=True,ax=None,xaxis='model_number',xmin=None,xmax=None,yrng=[-3.0,1.0],
					cmap=plt.cm.gist_ncar,num_labels=3,xlabel=None,points=False,abun_random=False,
				fig=None,fx=None,fy=None,minMod=-1,maxMod=-1,
				show_title_name=False,annotate_line=True,linestyle='-',colors=None,show_core=False,
				y2=None,y2rng=[None,None],fy2=None,y2Textcol=None,y2label=None,y2rev=False,y2log=False,y2col='k',xlog=False,xrev=False):
		
		fig,ax,modelIndex=self._setupHist(fig,ax,m,minMod,maxMod)
		
		x,xrngL,mInd=self._setXAxis(m.hist.data[xaxis][modelIndex],xmin,xmax,fx)

			
		abun_list=self._listAbun(m.hist,prefix='center_')
		num_plots=len(abun_list)
		
		if abun_random:
			random.shuffle(abun_list)
				
		self._cycleColors(ax,colors,cmap,num_plots)
			
		for i in abun_list:
			self._plotAnnotatedLine(ax=ax,x=x,y=m.hist.data[i],fy=fy,xmin=xrngL[0],
									xmax=xrngL[1],ymin=yrng[0],ymax=yrng[1],
									annotate_line=annotate_line,label=self.safeLabel(None,i,'center'),
									points=points,ylog=True,num_labels=num_labels,xlog=xlog,xrev=xrev)
			
		if y2 is not None:
			self._plotY2(fig,ax,x,m.hist.data,xrngL,xlog,xrev,mInd,y2,y2rng,fy2,y2Textcol,y2label,y2rev,y2log,y2col,points)
			
		if show_core:
			self._showMassLocHist(m,fig,ax,x,y,mInd)
		
		self._setXLabel(fig,ax,xlabel,xaxis)
		ax.set_ylabel(self.labels('Abundance'))

		if show:
			plt.show()

	def plotDynamo(self,m,xaxis='mass',model=None,show=True,ax=None,xmin=None,xmax=None,xlabel=None,y1rng=None,y2rng=None,
					show_burn=False,show_mix=False,legend=True,annotate_line=True,fig=None,fx=None,fy=None,
				show_title_name=False,show_title_model=False,show_title_age=False,show_rotation=True,show_shock=False):


		fig,ax=self._setupProf(fig,ax,m,model)
			
		ax1_1=fig.add_subplot(231)
		ax1_2=fig.add_subplot(234)
		
		ax2_t1=fig.add_subplot(233)
		ax2_t2=fig.add_subplot(236)
		
		ax2_1=ax2_t1.twinx()
		ax2_2=ax2_t2.twinx()
		
		for i in [ax1_1,ax1_2,ax2_1,ax2_2,ax2_t1,ax2_t2]:
			i.spines['top'].set_visible(False)
			i.spines['right'].set_visible(False)
			i.spines['bottom'].set_visible(False)
			i.spines['left'].set_visible(False)
			i.yaxis.set_major_locator(plt.NullLocator())
			i.xaxis.set_major_locator(plt.NullLocator())
			i.yaxis.set_minor_locator(plt.NullLocator())
			i.xaxis.set_minor_locator(plt.NullLocator())
			i.patch.set_facecolor('None')
		
		
		ax1_1.plot(0,0,color='w')
		ax1_2.plot(0,0,color='w')
		ax2_1.plot(0,0,color='w')
		ax2_2.plot(0,0,color='w')
	
		ax2=ax.twinx()
			
		x,xrngL,mInd=self._setXAxis(m.prof.data[xaxis],xmin,xmax,fx)
		
		#ind=(m.prof.data['dynamo_log_B_r']>-90)
		ax.plot(m.prof.data[xaxis],m.prof.data['dynamo_log_B_r'],label=r'$B_r$',linewidth=2,color='g')
		#ind=mInd&(m.prof.data['dynamo_log_B_phi']>-90)
		ax.plot(m.prof.data[xaxis],m.prof.data['dynamo_log_B_phi'],label=r'$B_{\phi}$',linewidth=2,color='b')
		
		if show_rotation:
			ax2.plot(m.prof.data[xaxis],np.log10(m.prof.data['omega']),'--',label=r'$\log_{10} \omega$',linewidth=2,color='r')
		#ind=mInd&(m.prof.data['dynamo_log_B_phi']>-90)
			ax2.plot(m.prof.data[xaxis],np.log10(m.prof.data['j_rot'])-20.0,'--',label=r'$\log_{10} j [10^{20}]$',linewidth=2,color='k')


		scale=2.1
		ax1_1.set_ylabel(r'$B_r$',color='g', labelpad=scale*mpl.rcParams['font.size'])
		ax1_2.set_ylabel(r'$B_{\phi}$',color='b', labelpad=scale*mpl.rcParams['font.size'])
		
		if show_rotation:
			ax2_1.set_ylabel(r'$\log_{10} \omega$',color='r', labelpad=scale*mpl.rcParams['font.size'])
			ax2_2.set_ylabel(r'$\log_{10} j [10^{20}]$',color='k', labelpad=scale*mpl.rcParams['font.size'])


		if show_burn:
			self._plotBurnRegions(m,ax,m.prof.data[xaxis],m.prof.data['dynamo_log_B_phi'],show_line=False,show_x=True,ind=mInd)

		if show_mix:
			self._plotMixRegions(m,ax,m.prof.data[xaxis],m.prof.data['dynamo_log_B_phi'],show_line=False,show_x=True,ind=mInd)

		if show_shock:
			self._showShockLoc(m.prof,fig,ax,x,yrng,mInd)
			
		self._setXLabel(fig,ax,xlabel,xaxis)
		self._setTicks(ax)
		self._setTicks(ax2)
		ax.set_xlim(xrngL)
		self._setYLim(ax,ax.get_ylim(),y1rng)
		self._setYLim(ax2,ax2.get_ylim(),y2rng)
		self.setTitle(ax,show_title_name,show_title_model,show_title_age,'Dynamo',m.prof.head["model_number"],m.prof.head["star_age"])
		
		
		if show:
			plt.show()

	def plotAngMom(self,m,xaxis='mass',model=None,show=True,ax=None,xmin=None,xmax=None,xlabel=None,yrng=[0.0,10.0],
					show_burn=False,show_mix=False,legend=True,annotate_line=True,num_labels=5,fig=None,fx=None,fy=None,
				show_title_name=False,show_title_model=False,show_title_age=False,points=False,show_core=False,show_shock=False,
				y2=None,y2rng=[None,None],fy2=None,y2Textcol=None,y2label=None,y2rev=False,y2log=False,y2col='k',xlog=False,xrev=False):
		
		fig,ax=self._setupProf(fig,ax,m,model)
			
		x,xrngL,mInd=self._setXAxis(m.prof.data[xaxis],xmin,xmax,fx)

		for i in m.prof.data_names:         
			if "am_log_D" in i:
				px,py=self._plotAnnotatedLine(ax=ax,x=x,y=m.prof.data[i],fy=fy,xmin=xrngL[0],xmax=xrngL[1],
										ymin=yrng[0],ymax=yrng[1],annotate_line=annotate_line,
										label=r"$D_{"+i.split('_')[3]+"}$",points=points,
										ylog=True,num_labels=num_labels,xrev=xrev,xlog=log)

		if show_burn:
			self._plotBurnRegions(m,ax,px,m.prof.data[i],show_line=False,show_x=True,ind=mInd)

		if show_mix:
			self._plotMixRegions(m,ax,px,m.prof.data[i],show_line=False,show_x=True,ind=mInd)
			
		if show_core:
			self._showMassLocHist(m,fig,ax,x,y,mInd)
			
		if show_shock:
			self._showShockLoc(m.prof,fig,ax,x,yrng,mInd)
			
		if y2 is not None:
			self._plotY2(fig,ax,x,m.prof.data,xrngL,xlog,xrev,mInd,y2,y2rng,fy2,y2Textcol,y2label,y2rev,y2log,y2col,points)

		if legend:
			ax.legend(loc=0)

		self._setXLabel(fig,ax,xlabel,xaxis)
		self.setTitle(ax,show_title_name,show_title_model,show_title_age,'Ang mom',m.prof.head["model_number"],m.prof.head["star_age"])
		
		
		if show:
			plt.show()
			
	def plotBurn(self,m,xaxis='mass',model=None,show=True,ax=None,xmin=None,xmax=None,xlabel=None,
				cmap=plt.cm.gist_ncar,yrng=[0.0,10.0],num_labels=7,burn_random=False,points=False,
				show_burn=False,show_mix=False,fig=None,fx=None,fy=None,
				show_title_name=False,show_title_model=False,show_title_age=False,annotate_line=True,show_shock=False,
				y2=None,y2rng=[None,None],fy2=None,y2Textcol=None,y2label=None,y2rev=False,y2log=False,y2col='k',xlog=False,xrev=False):
		
		fig,ax=self._setupProf(fig,ax,m,model)
			
		x,xrngL,mInd=self._setXAxis(m.prof.data[xaxis],xmin,xmax,fx)


		burn_list=self._listBurn(m.prof)
		num_plots=len(burn_list)
		
		if burn_random:
			random.shuffle(burn_list)
				
		self._cycleColors(ax,None,cmap,num_plots)
			
		for i in burn_list:
			px,py=self._plotAnnotatedLine(ax=ax,x=x,y=m.prof.data[i],fy=fy,xmin=xrngL[0],
									xmax=xrngL[1],ymin=yrng[0],ymax=yrng[1],annotate_line=annotate_line,
									label=self.safeLabel(None,i),points=points,ylog=True,num_labels=num_labels,xrev=xrev,xlog=xlog)

		
		if show_burn:
			self._plotBurnRegions(m,ax,px,py,show_line=False,show_x=True,ind=mInd)

		if show_mix:
			self._plotMixRegions(m,ax,px,py,show_line=False,show_x=True,ind=mInd)
			
		if show_shock:
			self._showShockLoc(m.prof,fig,ax,x,yrng,mInd)
			
		if y2 is not None:
			self._plotY2(fig,ax,x,m.prof.data,xrngL,xlog,xrev,mInd,y2,y2rng,fy2,y2Textcol,y2label,y2rev,y2log,y2col,points)
		
		self._setXLabel(fig,ax,xlabel,xaxis)
		self.setTitle(ax,show_title_name,show_title_model,show_title_age,'Burn',m.prof.head["model_number"],m.prof.head["star_age"])
		
		
		if show:
			plt.show()
			
	def plotMix(self,m,xaxis='mass',model=None,show=True,ax=None,xmin=None,xmax=None,xlabel=None,
				cmap=plt.cm.gist_ncar,yrng=[0.0,5.0],num_labels=7,mix_random=False,points=False,
				show_burn=False,fig=None,fx=None,fy=None,
				show_title_name=False,show_title_model=False,show_title_age=False,annotate_line=True,show_shock=False,colors=None,
				y2=None,y2rng=[None,None],fy2=None,y2Textcol=None,y2label=None,y2rev=False,y2log=False,y2col='k',xlog=False,xrev=False):
		
		fig,ax=self._setupProf(fig,ax,m,model)
			
		x,xrngL,mInd=self._setXAxis(m.prof.data[xaxis],xmin,xmax,fx)

		mix_list=self._listMix(m.prof)
		num_plots=len(mix_list)
		
		if mix_random:
			random.shuffle(mix_list)
				
		self._cycleColors(ax,None,cmap,num_plots)
			
		for i in mix_list:
			px,py=self._plotAnnotatedLine(ax=ax,x=x,y=m.prof.data[i],fy=fy,xmin=xrngL[0],
									xmax=xrngL[1],ymin=yrng[0],ymax=yrng[1],
									annotate_line=annotate_line,label=i.split('_')[2],
									points=points,ylog=False,num_labels=num_labels,xlog=xlog,xrev=xrev)
		
		if show_burn:
			self._plotBurnRegions(m,ax,px,py,show_line=False,show_x=True,ind=mInd)
			
		if show_shock:
			self._showShockLoc(m.prof,fig,ax,x,yrng,mInd)
			
		if y2 is not None:
			self._plotY2(fig,ax,x,m.prof.data,xrngL,xlog,xrev,mInd,y2,y2rng,fy2,y2Textcol,y2label,y2rev,y2log,y2col,points)
		
		self._setXLabel(fig,ax,xlabel,xaxis)
		self.setTitle(ax,show_title_name,show_title_model,show_title_age,'Mixing',m.prof.head["model_number"],m.prof.head["star_age"])
		
		
		if show:
			plt.show()

	def plotBurnSummary(self,m,xaxis='model_number',minMod=0,maxMod=-1,show=True,ax=None,xmin=None,xmax=None,xlabel=None,
				cmap=plt.cm.nipy_spectral,yrng=[0.0,10.0],num_labels=7,burn_random=False,points=False,
				show_burn=False,show_mix=False,fig=None,fx=None,fy=None,annotate_line=True,show_core=False,
				y2=None,y2rng=[None,None],fy2=None,y2Textcol=None,y2label=None,y2rev=False,y2log=False,y2col='k',xlog=False,xrev=False):
		
		fig,ax,modelIndex=self._setupHist(fig,ax,m,minMod,maxMod)
		
		x,xrngL,mInd=self._setXAxis(m.hist.data[xaxis][modelIndex],xmin,xmax,fx)

			
		burn_list=self._listBurnHistory(m.prof)
		num_plots=len(burn_list)
		
		if burn_random:
			random.shuffle(burn_list)
				
		self._cycleColors(ax,colors,cmap,num_plots)
			
		for i in burn_list:
			self._plotAnnotatedLine(ax=ax,x=x,y=m.prof.data[i],fy=fy,xmin=xrngL[0],
									xmax=xrngL[1],ymin=yrng[0],ymax=yrng[1],
									annotate_line=annotate_line,label=self.safeLabel(None,i),
									points=points,ylog=True,num_labels=num_labels,xrev=xrev,xlog=xlog)

		if show_burn:
			self._plotBurnRegions(m,ax,x[mInd],y,show_line=False,show_x=True,ind=mInd)

		if show_mix:
			self._plotMixRegions(m,ax,x[mInd],y,show_line=False,show_x=True,ind=mInd)
			
		if y2 is not None:
			self._plotY2(fig,ax,x,m.hist.data,xrngL,xlog,xrev,mInd,y2,y2rng,fy2,y2Textcol,y2label,y2rev,y2log,y2col,points)
			
		if show_core:
			self._showMassLocHist(m,fig,ax,x,y,mInd)
		
		self._setXLabel(fig,ax,xlabel,xaxis)
		ax.set_ylabel(self.labels('log_lum'))

		if show:
			plt.show()

	def plotAbunSummary(self,m,xaxis='model_number',minMod=0,maxMod=-1,show=True,ax=None,xmin=None,xmax=None,xlabel=None,
				cmap=plt.cm.nipy_spectral,yrng=[0.0,10.0],num_labels=7,abun_random=False,points=False,
				show_burn=False,show_mix=False,abun=None,fig=None,fx=None,fy=None,annotate_line=True,linestyle='-',colors=None,
				show_core=False,
				y2=None,y2rng=[None,None],fy2=None,y2Textcol=None,y2label=None,y2rev=False,y2log=False,y2col='k',xlog=False,xrev=False):
		
		fig,ax,modelIndex=self._setupHist(fig,ax,m,minMod,maxMod)
		
		x,xrngL,mInd=self._setXAxis(m.hist.data[xaxis][modelIndex],xmin,xmax,fx)

			
		if abun is None:
			abun_list,log=self._listAbun(m.hist)
		else:
			abun_list=abun
			
		num_plots=len(abun_list)
		#Helps when we have many elements not on the plot that stretch the colormap
		if abun_random:
			random.shuffle(abun_list)
		
		self._cycleColors(ax,colors,cmap,num_plots)
			
		for i in abun_list:
			y=m.hist.data["log_total_mass_"+i][mInd]
			self._plotAnnotatedLine(ax=ax,x=x,y=y,fy=fy,xmin=xrngL[0],
									xmax=xrngL[1],ymin=yrng[0],ymax=yrng[1],
									annotate_line=annotate_line,label=self.safeLabel(None,i),
									points=points,ylog=True,num_labels=num_labels,linestyle=linestyle,
									xrev=xrev,xlog=xlog)


		if show_burn:
			self._plotBurnRegions(m,ax,x[mInd],y,show_line=False,show_x=True,ind=mInd)

		if show_mix:
			self._plotMixRegions(m,ax,x[mInd],y,show_line=False,show_x=True,ind=mInd)
			
		if y2 is not None:
			self._plotY2(fig,ax,x,m.hist.data,xrngL,xlog,xrev,mInd,y2,y2rng,fy2,y2Textcol,y2label,y2rev,y2log,y2col,points)
			
		if show_core:
			self._showMassLocHist(m,fig,ax,x,y,mInd)
		
		self._setXLabel(fig,ax,xlabel,xaxis)
		ax.set_ylabel(self.labels('log_abundance'))

		if show:
			plt.show()


	def plotProfile(self,m,model=None,xaxis='mass',y1='logT',y2=None,show=True,ax=None,xmin=None,xmax=None,
					xlog=False,y1log=False,y2log=False,y1col='b',
					y2col='r',xrev=False,y1rev=False,y2rev=False,points=False,xlabel=None,y1label=None,y2label=None,
					show_burn=False,show_burn_2=False,show_burn_x=False,show_burn_line=False,
					show_mix=False,show_mix_2=False,show_mix_x=False,show_mix_line=False,
					y1Textcol=None,y2Textcol=None,fig=None,y1rng=[None,None],y2rng=[None,None],
					fx=None,fy1=None,fy2=None,
					show_title_name=False,title_name=None,show_title_model=False,show_title_age=False,
					y1linelabel=None,show_core_loc=False,show_shock=False,yrng=None):
		
		fig,ax=self._setupProf(fig,ax,m,model)

		x,xrngL,mInd=self._setXAxis(m.prof.data[xaxis],xmin,xmax,fx)
		
		y=m.prof.data[y1][mInd]
		px,py=self._plotAnnotatedLine(ax=ax,x=x[mInd],y=y,fy=fy1,xmin=xrngL[0],xmax=xrngL[1],
								ymin=y1rng[0],ymax=y1rng[1],annotate_line=False,
								label=self.safeLabel(y1label,y1),points=points,
								xlog=xlog,ylog=y1log,xrev=xrev,yrev=y1rev,linecol=y1col)
		
		if y1Textcol is None:
			y1labcol=y1col
		else:
			y1labcol=y1Textcol


		ax.set_ylabel(self.safeLabel(y1label,y1), color=y1labcol)
		if yrng is None:
			self._setYLim(ax,ax.get_ylim(),y1rng,rev=y1rev,log=y1log)
		else:
			self._setYLim(ax,ax.get_ylim(),yrng,rev=y1rev,log=y1log)

		if show_burn:
			self._plotBurnRegions(m,ax,px,py,show_line=show_burn_line,show_x=show_burn_x,ind=mInd)

		if show_mix:
			self._plotMixRegions(m,ax,px,py,show_line=show_mix_line,show_x=show_mix_x,ind=mInd)
	
		if show_burn or show_mix:
			self._showBurnMixLegend(ax,burn=show_burn,mix=show_mix)

		if show_core_loc:
			self._plotCoreLoc(m.prof,ax,xaxis,px,ax.get_ylim()[0],ax.get_ylim()[1])
	
		if show_shock:
			self._showShockLoc(m.prof,fig,ax,x,ax.get_ylim(),mInd)
	
		if y2 is not None:
			self._plotY2(fig,ax,x,m.prof.data,xrngL,xlog,xrev,mInd,y2,y2rng,fy2,y2Textcol,y2label,y2rev,y2log,y2col,points)

		self._setTicks(ax)
	
		self._setXLabel(fig,ax,xlabel,xaxis)
		self._setYLabel(fig,ax,y1label,y1,y1col)
		
		self.setTitle(ax,show_title_name,show_title_model,show_title_age,title_name,m.prof.head["model_number"],m.prof.head["star_age"])
		
		
		if show:
			plt.show()

	def plotHistory(self,m,xaxis='model_number',y1='star_mass',y2=None,show=True,
					ax=None,xmin=None,xmax=None,xlog=False,y1log=False,
					y2log=False,y1col='b',y2col='r',minMod=0,maxMod=-1,xrev=False,
					y1rev=False,y2rev=False,points=False,xlabel=None,y1label=None,
					y2label=None,fig=None,y1rng=[None,None],y2rng=[None,None],
					fx=None,fy1=None,fy2=None,show_core=False,y1Textcol=None,y2Textcol=None):
		
		fig,ax,modelIndex=self._setupHist(fig,ax,m,minMod,maxMod)
		
		x,xrngL,mInd=self._setXAxis(m.hist.data[xaxis][modelIndex],xmin,xmax,fx)
			
		y=m.hist.data[y1][modelIndex][mInd]
		self._plotAnnotatedLine(ax=ax,x=x[mInd],y=y,fy=fy1,xmin=xrngL[0],xmax=xrngL[1],
								ymin=y1rng[0],ymax=y1rng[1],annotate_line=False,
								label=self.safeLabel(y1label,y1),points=points,
								xlog=xlog,ylog=y1log,xrev=xrev,yrev=y1rev,linecol=y1col)
		
		self._setYLim(ax,ax.get_ylim(),y1rng,rev=y1rev,log=y1log)


		if y2 is not None:
			self._plotY2(fig,ax,x,m.hist.data,xrngL,xlog,xrev,mInd,y2,y2rng,fy2,y2Textcol,y2label,y2rev,y2log,y2col,points)

		self._setXLabel(fig,ax,xlabel,xaxis)
		self._setYLabel(fig,ax,y1label,y1,y1col)
			
		if show_core:
			self._showMassLocHist(m,fig,ax,x,y,mInd)
		
		if show:
			plt.show()

	def plotKip(self,m,show=True,reloadHistory=False,xaxis='num',ageZero=0.0,ax=None,xrng=[-1,-1],mix=None,
				cmin=None,cmax=None,burnMap=[mpl.cm.Purples_r,mpl.cm.hot_r],fig=None,yrng=None,
				show_mass_loc=False,show_mix_labels=True,mix_alpha=1.0,step=1,y2=None,title=None,y2rng=None):
					
		if fig==None:
			fig=plt.figure(figsize=(12,12))
		
		if title is not None:
			fig.suptitle(title)
			
		if show_mix_labels:
			self._addMixLabelsAxis(fig)

		if ax==None:
			ax=fig.add_subplot(111)
			
		if y2 is not None:
			ax2=ax.twinx()
		
		if reloadHistory:
			m.loadHistory()
			
		try:
			xx=m.hist.data['model_number']
		except KeyError:
			raise ValueError("Must call loadHistory first")
		
		modInd=np.zeros(np.size(m.hist.data["model_number"]),dtype='bool')
		modInd[::step]=True
		
		if xrng[0]>=0:
			modInd=modInd&(m.hist.data["model_number"]>=xrng[0])&(m.hist.data["model_number"]<=xrng[1])
			

		if np.count_nonzero(modInd) > 40000:
			print("Warning attempting to plot more than 40,000 models")
			print("This may take a long time")

		if 'm_center' in m.hist.data.dtype.names:
			if np.any(m.hist.data['m_center'] > m.hist.star_mass):
				#m_center in grams
				m_center=m.hist.data['m_center']/self.msun
			else:
				#Solar units
				m_center=m.hist.data['m_center']
		else:
			m_center=np.zeros(np.size(m.hist.data['model_number']))
			
			
		q=np.linspace(np.min(m_center),np.max(m.hist.data["star_mass"]),np.max(m.hist.data["num_zones"][modInd]))
		numModels=np.count_nonzero(modInd)

		numMixZones=int([x.split('_')[2] for  x in m.hist.data.dtype.names if "mix_qtop" in x][-1])
		numBurnZones=int([x.split('_')[2] for x in m.hist.data.dtype.names if "burn_qtop" in x][-1])

		burnZones=np.zeros((numModels,np.size(q)))
		burnZones[:,:]=np.nan
		k=0		
		for jj in m.hist.data["model_number"][modInd]:
			ind2b=np.zeros(np.size(q),dtype='bool')
			i=m.hist.data["model_number"]==jj
			ind2b=(q<=m_center[i])
			for j in range(1,numBurnZones+1):
				indb=(q<=m_center[i]+m.hist.data["burn_qtop_"+str(j)][i]*m.hist.data['star_mass'][i])&np.logical_not(ind2b)
				burnZones[k,indb]=m.hist.data["burn_type_"+str(j)][i]
				ind2b=ind2b|indb
				if m.hist.data["burn_qtop_"+str(j)][i] ==1.0:
					break
			k=k+1

		Xmin=m.hist.data["model_number"][modInd][0]
		Xmax=m.hist.data["model_number"][modInd][-1]
      
		Ymin=q[0]
		Ymax=q[-1]
		extent=(Xmin,Xmax,Ymin,Ymax)
		
		burnZones[burnZones<-100]=0.0

		if cmin is None:
			vmin=np.nanmin(burnZones)
		else:
			vmin=cmin
			
		if cmax is None:
			vmax=np.nanmax(burnZones)
		else:
			vmax=cmax
			
		if vmin < 0:
			vmax=np.maximum(np.abs(vmax),np.abs(vmin))
			vmin=-vmax
			newCm=self.mergeCmaps(burnMap,[[0.0,0.5],[0.5,1.0]])
		else:
			vmin=0
			newCm=burnMap[-1]

		im1=ax.imshow(burnZones.T,cmap=newCm,extent=extent,
				interpolation='nearest',
				origin='lower',aspect='auto',
				vmin=vmin,
				vmax=vmax)		
		burnZones=0

		
		mixZones=np.zeros((numModels,np.size(q)))
		mixZones[:,:]=-np.nan
		k=0
		for jj in m.hist.data["model_number"][modInd]:
			ind2=np.zeros(np.size(q),dtype='bool')
			i=m.hist.data["model_number"]==jj
			ind2b=(q<=m_center[i])
			for j in range(1,numMixZones+1):
				ind=(q<= m_center[i]+m.hist.data["mix_qtop_"+str(j)][i]*m.hist.data['star_mass'][i])&np.logical_not(ind2)
				if mix is None:
					mixZones[k,ind]=m.hist.data["mix_type_"+str(j)][i]
				elif mix ==-1 :
					mixZones[k,ind]=0.0
				elif m.hist.data["mix_type_"+str(j)][i] in mix:
					mixZones[k,ind]=m.hist.data["mix_type_"+str(j)][i]
				else:
					mixZones[k,ind]=0.0
				ind2=ind2|ind
				if m.hist.data["mix_qtop_"+str(j)][i]==1.0:
					break
			k=k+1		

					
		mixZones[mixZones==0]=-np.nan
		
		mixCmap,mixNorm=self._setMixRegionsCol(kip=True)
		
		ax.imshow(mixZones.T,cmap=mixCmap,norm=mixNorm,extent=extent,interpolation='nearest',origin='lower',aspect='auto',alpha=mix_alpha)
		mixZones=0
		ax.set_xlabel(r"$\rm{Model\; number}$")
		ax.set_ylabel(r"$\rm{Mass}\; [M_{\odot}]$")
		
		cb=fig.colorbar(im1,ax=ax)
		cb.solids.set_edgecolor("face")

		cb.set_label(r'$\rm{sign}\left(\epsilon_{\rm{nuc}}-\epsilon_{\nu}\right)\log_{10}\left(\rm{max}\left(1.0,|\epsilon_{\rm{nuc}}-\epsilon_{\nu}|\right)\right)$')

		self._setYLim(ax,ax.get_ylim(),yrng)
		
		#Add line at outer mass location and inner
		ax.plot(m.hist.data['model_number'][modInd],m.hist.data['star_mass'][modInd],color='k')
		ax.plot(m.hist.data['model_number'][modInd],m_center[modInd],color='k')
		
		
		if y2 is not None:
			# Update axes 2 locations after ax1 is moved by the colorbar
			ax2.set_position(ax.get_position())
			ax2.plot(m.hist.data['model_number'][modInd],m.hist.data[y2][modInd],color='k')
			if y2rng is not None:
				ax2.set_ylim(y2rng)
		
		if show_mass_loc:
			self._showMassLoc(m,fig,ax,np.linspace(Xmin,Xmax,np.count_nonzero(modInd)),modInd)
		
		if show:
			plt.show()
		
	def plotKip2(self,m,show=True,reloadHistory=False,xaxis='num',ageZero=0.0,ax=None,xrng=[-1,-1],mix=None,
				cmin=None,cmax=None,burnMap=[mpl.cm.Purples_r,mpl.cm.hot_r],fig=None,yrng=None,
				show_mass_loc=False,show_mix_labels=True,mix_alpha=1.0,step=1,max_mass=99999.0,age_collapse=False,age_log=True,age_reverse=False,
				mod_out=None,megayears=False,xlabel=None,title=None,colorbar=True,burn=True,end_time=None,ylabel=None,age_zero=None,
				num_x=None,num_y=None,y2=None):
					
		if fig==None:
			fig=plt.figure(figsize=(12,12))
			
		if title is not None:
			fig.suptitle(title)
			
		if show_mix_labels:
			self._addMixLabelsAxis(fig)

		if ax==None:
			ax=fig.add_subplot(111)
		
		if reloadHistory:
			m.loadHistory()
			
		try:
			xx=m.hist.data['model_number']
		except:
			raise ValueError("Must call loadHistory first")
		
		modInd=np.zeros(np.size(m.hist.data["model_number"]),dtype='bool')
		modInd[:]=True
		
		modInd[::step]=False
		modInd[:]=np.logical_not(modInd)
		
		if mod_out is not None:
			modInd=modInd&mod_out
			
		if xrng[0]>=0:
			modInd=modInd&(m.hist.data["model_number"]>=xrng[0])&(m.hist.data["model_number"]<=xrng[1])
		
		#Age in years does not have enough digits to be able to distingush the final models in pre-sn progenitors
		age=np.cumsum(10**np.longdouble(m.hist.log_dt))
		
		if age_collapse:
			xx=age[-1]
			if end_time is not None:
				xx=end_time
			age=xx-age
			#Fudge the last value not to be exactly 0.0
			age[-1]=(age[-2]/2.0)
			
		if age_zero is not None:
			age=age_zero-age
			#Fudge the first value not to be exactly 0.0
			age[0]=age[1]/2.0
		
		modInd2=np.zeros(np.shape(modInd),dtype='bool')
		modInd2[0]=modInd[0]
		modInd2[1:]=(np.diff(age)!=0.0)
		modInd=modInd&modInd2
		
		age=age[modInd]
		
		if megayears:
			age=age/10**6
		
		if age_log:
			age=np.log10(age)
		
		if age_reverse:
			age=age[::-1]
			
		q=np.linspace(0.0,np.minimum(max_mass,np.max(m.hist.data["star_mass"])),np.max(m.hist.data["num_zones"][modInd]))
		numModels=np.count_nonzero(modInd)
		
		if num_x is None:
			num_x=numModels
			
		if num_x<numModels:
			print("Not supported for num_x< numModels, leave num_x as None or bigger than numModels")
			return
		
		lin_age=np.linspace(age[0],age[-1],num_x)
		
		if num_y is None:
			num_y=np.size(q)
      

		burnZones=np.zeros((numModels,np.size(q)))
			
		self.numMixZones=int([x.split('_')[2] for  x in m.hist.data.dtype.names if "mix_qtop" in x][-1])
		self.numBurnZones=int([x.split('_')[2] for x in m.hist.data.dtype.names if "burn_qtop" in x][-1])

		k=0		
		for jj in m.hist.data["model_number"][modInd]:
			ind2b=np.zeros(np.size(q),dtype='bool')
			i=m.hist.data["model_number"]==jj
			for j in range(1,self.numBurnZones+1):
				indb=(q<= m.hist.data["burn_qtop_"+str(j)][i]*m.hist.data['star_mass'][i])&np.logical_not(ind2b)
				burnZones[k,indb]=m.hist.data["burn_type_"+str(j)][i]
				ind2b=ind2b|indb
			k=k+1

		Xmin=m.hist.data["model_number"][modInd][0]
		Xmax=m.hist.data["model_number"][modInd][-1]
			
		Ymin=q[0]
		Ymax=q[-1]
		
		burnZones[burnZones<-100]=0.0
		
		extent=(age[0],age[-1],Ymin,Ymax)
		extent=np.double(np.array(extent))

		if cmin is None:
			vmin=np.nanmin(burnZones)
		else:
			vmin=cmin
			
		if cmax is None:
			vmax=np.nanmax(burnZones)
		else:
			vmax=cmax
			
		if vmin < 0:
			vmax=np.maximum(np.abs(vmax),np.abs(vmin))
			vmin=-vmax
			newCm=self.mergeCmaps(burnMap,[[0.0,0.5],[0.5,1.0]])
		else:
			vmin=0
			newCm=burnMap[-1]
		
		if burn:
			burnZones2=np.zeros(np.shape(burnZones))
			sorter=np.argsort(age)
			ind=np.searchsorted(age,lin_age,sorter=sorter)
			burnZones2[:,:]=burnZones[sorter[ind],:]
			im1=ax.imshow(burnZones2.T,cmap=newCm,extent=extent,interpolation='nearest',origin='lower',aspect='auto',vmin=vmin,vmax=vmax)
         
		mixCmap,mixNorm=self._setMixRegionsCol(kip=True)
					
				
		if mix != -1:
			mixZones=np.zeros((numModels,np.size(q)))
			k=0
			for jj in m.hist.data["model_number"][modInd]:
				ind2=np.zeros(np.size(q),dtype='bool')
				i=m.hist.data["model_number"]==jj
				for j in range(1,self.numMixZones+1):
					ind=(q<= m.hist.data["mix_qtop_"+str(j)][i]*m.hist.data['star_mass'][i])&np.logical_not(ind2)
					if mix is None:
						mixZones[k,ind]=m.hist.data["mix_type_"+str(j)][i]
					elif mix ==-1 :
						mixZones[k,ind]=0.0
					elif m.hist.data["mix_type_"+str(j)][i] in mix:
						mixZones[k,ind]=m.hist.data["mix_type_"+str(j)][i]
					else:
						mixZones[k,ind]=0.0
					ind2=ind2|ind
				k=k+1		
			
			mixZones2=np.zeros(np.shape(mixZones))
			sorter=np.argsort(age)
			ind=np.searchsorted(age,lin_age,sorter=sorter)
			mixZones2[:,:]=mixZones[sorter[ind],:]
			mixZones2[mixZones2<1]=np.nan
			ax.imshow(mixZones2.T,cmap=mixCmap,norm=mixNorm,extent=extent,interpolation='nearest',origin='lower',aspect='auto',alpha=mix_alpha)
			
		if ylabel is not None:
			ax.set_ylabel(ylabel)
		else:
			ax.set_ylabel(r"$\rm{Mass}\; [M_{\odot}]$")
		
		if colorbar and burn:
			cb=fig.colorbar(im1)
			cb.solids.set_edgecolor("face")

			cb.set_label(r'$\rm{sign}\left(\epsilon_{\rm{nuc}}-\epsilon_{\nu}\right)\log_{10}\left(\rm{max}\left(1.0,|\epsilon_{\rm{nuc}}-\epsilon_{\nu}|\right)\right)$')
			fig.set_size_inches(12,9.45)
				
		##Add line at outer mass location
		f = interp1d(age, m.hist.data['star_mass'][modInd])
		ax.plot(lin_age,f(lin_age),c='k')
		
		if y2 is not None:
			ax2=ax.twinx()
			f = interp1d(age, m.hist.data[y2][modInd])
			ax2.plot(lin_age,f(lin_age),c='k')
		
		if xlabel is None:
			if age_log:
				if age_collapse:
					ax.set_xlabel(r"$\log_{10}\;\left(\rm{\tau_{cc}-\tau}\right)\; [\rm{yr}]$")
				else:
					ax.set_xlabel(r"$\log\; \left(\tau/\rm{Myr}\right)$")
			else:
				if age_collapse:
					if megayears:
						ax.set_xlabel(r"$\left(\rm{\tau_{cc}-\tau}\right)\; [\rm{Myr}]$")
					else:
						ax.set_xlabel(r"$\left(\rm{\tau_{cc}-\tau}\right)\; [\rm{yr}]$")
				else:
					ax.set_xlabel(r"$\tau\; \left(\rm{Myr}\right)$")
		else:
			ax.set_xlabel(xlabel)
	
		if show_mass_loc:
			self._showMassLoc(m,fig,ax,np.linspace(Xmin,Xmax,np.count_nonzero(modInd)),modInd)
	
		self._setYLim(ax,ax.get_ylim(),yrng)
		
		self._setTicks(ax)
		
		if show:
			plt.show()
			
			
	#Will replace plotKip and plotKip2 when finished
	#def plotKip3(self,m,plot_type='hist',xaxis='model_number',yaxis='mass',
				#xrng=[-1,-1],yrng=[-1,-1],x_index=None,y_index=None,xtsep=1,ystep=1,
				#xlabel=None,ylabel=None,title=None,
				#show=True,reloadHistory=False,ax=None,fig=None,title=None
				#show_mix=True,mix=None,show_burn=True,
				#cmin=None,cmax=None,colormap=None,burnMap=[mpl.cm.Purples_r,mpl.cm.hot_r],colorbar=True,
				#show_mass_loc=False,show_mix_labels=True,mix_alpha=1.0,
				#age_collapse=False,age_log=True,age_reverse=False,age_units='yrs',end_time=None,age_zero=None
				#num_x=None,num_y=None,y2=None):
					
		#if fig==None:
			#fig=plt.figure(figsize=(12,12))
			
		#if title is not None:
			#fig.suptitle(title)
			
			
		#if plot_type=='hist' and show_mix_labels	
			#self._addMixLabelsAxis(fig)

		#if ax==None:
			#ax=fig.add_subplot(111)
		
		#if plot_type=='history':			
			#try:
				#model_num=m.hist.data['model_number']
			#except AttributeError:
				#raise ValueError("Must call loadHistory first")
		#elif plot_type=='profile'
			#try:
				#model_num=m.prof.head['model_number']
			#except AttributeError:
				#raise ValueError("Must load a profile file first")
				
			#try:
				#y=m.prof.data[yaxis]
			#except AttributeError:
				#raise ValueError("No value "+yaxis+" found")
							
		#else:
			#raise ValueError("plot_type must be either history or profile, got "+plot_type)	
	
	

		#if not (xaxis=='model_number' or xaxis=='age'):
			#raise ValueError("Kips can only plot model_number or age, got "+xaxis)

						
		##Extract Data
		#data_x=[]
		#data_y=[]
		#data_z=[]
		
		#if plot_type=='hist':
			#if xaxis=='model_number':
				#data_x=m.hist.model_number
			#else:
				#data_x=m.hist.star_age
		#else:
			
		
		
		
		
		
		
	def plotTRho(self,m,model=None,show=True,ax=None,xmin=-4.0,xmax=10.0,fig=None,yrng=[3.0,10.0],
				show_burn=False,show_mix=False,
				showAll=False,showBurn=False,showPgas=False,showDegeneracy=False,
				showGamma=False,showEOS=False,logT=False,logRho=False,
				ycol='k'):
		
		fig,ax=self._setupProf(fig,ax,m,model)
			
		try:
			x=m.prof.logRho
			xname='logRho'
			xlog=False
		except:
			x=m.prof.Rho
			xname='rho'
			xlog=True
			
		try:
			y=m.prof.logT
			yname='logT'
			ylog=False
		except:
			y=m.prof.temperature
			yname='temperature'
			ylog=True
	
		self.plotProfile(m,xaxis=xname,y1=yname,y1log=ylog,xlog=xlog,model=model,show=False,
               show_mix=show_mix,show_burn=show_burn,show_mix_line=True,show_burn_line=True,
               xmin=xmin,xmax=xmax,ax=ax,y1label=self.labels('teff',log=True),
               xlabel=self.labels('rho',log=True),fig=fig,y1rng=yrng,y2rng=None,y1col=ycol)

		if showBurn or showAll:
			self._showBurnData(ax)
		
		if showPgas or showAll:
			self._showPgas(ax)
			
		if showDegeneracy or showAll:
			self._showDegeneracy(ax)
			
		if showGamma or showAll:
			self._showGamma4(ax)
			
		if showEOS or showAll:
			self._showEOS(ax)

		if show:
			plt.show()
							
							
	def plotHR(self,m,minMod=0,maxMod=-1,show=True,ax=None,xmin=None,xmax=None,fig=None,points=None):
		self.plotHistory(m,xaxis='log_Teff',y1='log_L',y1log=False,minMod=minMod,
							maxMod=maxMod,show=show,xmin=xmin,xmax=xmax,xrev=True,y1rev=False,ax=ax,y1col='k',
							xlabel=self.labels('teff',log=True),y1label=self.labels('lum',log=True),
							fig=fig,points=points)
	
	def mergeCmaps(self,cmaps,rng=[[0.0,0.5],[0.5,1.0]]):
		"""
		Creates a diverging colomap
		
		cmaps: list of colormaps (ie [cm.Purples_r,cm.hot_r])
		rng: list of list with the rng to define the colormaps over ie [[0.0,0.5],[0.5,1.0]]
				to have a diverging colormap centered at the mid point
				
		Returns:
		LinearSegmentedColormap
		"""
		cdict={'red':[],'blue':[],'green':[]}
		for i in range(len(cmaps)):
			cmap=cmaps[i]
			minX=rng[i][0]
			maxX=rng[i][1]
			cmapseg=cmap._segmentdata
			for key in ('red','green','blue'):
				for j in range(len(cmapseg[key])):
					cdict[key].append([minX+(maxX-minX)*cmapseg[key][j][0],cmapseg[key][j][1],cmapseg[key][j][2]])

		return mpl.colors.LinearSegmentedColormap('colormap',cdict,1024)
		
	def stackedPlots(self,m,typ='profile',num=1,model=None,xaxis='mass',show=True,
						fig=None,ax=None,xmin=None,xmax=None,xlog=False,xlabel=None,
						xrev=False,y1rev=[],y2rev=[],points=False,minMod=0,maxMod=-1,
						y1=[],y2=[],y1log=[],y2log=[],y1col=[],
						y2col=[],y1label=[],y2label=[]):
		if num<2:
			raise(ValueError,'num must be >=2')
		
		empty=[None]*len(y1)
		f=[False]*len(y1)
		if len(y1)>0:
			if not y2:
				y2=empty
			if not y1log:
				y1L=f
			if not y2log:
				y2L=f
			if not y1rev:
				y1rev=f
			if not y2rev:
				y2rev=f
			if not y1col:
				y1col=['r']*len(y1)
			if not y2col:
				y2col=['b']*len(y1)
			if not y1label:
				y1label=empty*len(y1)
			if not y2label:
				y2label=empty*len(y1)
			
		f, axis = plt.subplots(num, sharex=True)
		f.subplots_adjust(hspace=0)
		
		for i in range(num):
			if typ=="profile":
				self.plotProfile(m=m,model=model,xaxis=xaxis,show=False,ax=axis[i],xmin=xmin,xmax=xmax,xL=xL,xlabel=xlabel,
							xrev=xrev,y1rev=y1rev[i],y2rev=y2rev[i],points=points,
							y1=y1[i],y2=y2[i],y1log=y1log[i],y2log=y2log[i],y1col=y1col[i],
							y2col=y2col[i],y1label=y1label[i],y2label=y2label[i])
			else:
				self.plotHistory(m=m,xaxis=xaxis,show=False,ax=axis[i],xmin=xmin,xmax=xmax,xL=xL,xlabel=xlabel,
							xrev=xrev,y1rev=y1rev[i],y2rev=y2rev[i],points=points,
							y1=y1[i],y2=y2[i],y1log=y1log[i],y2log=y2log[i],y1col=y1col[i],
							y2col=y2col[i],y1label=y1label[i],y2label=y2label[i],minMod=minMod,maxMod=maxMod)
		
		if show:
			plt.show()

	def plotMultiProfiles(self,m,mods=None,index=None,xaxis='mass',y1='',
					   show=True,ax=None,xmin=None,xmax=None,
					   xlog=False,y1log=False,
						cmap=plt.cm.gist_ncar,xrev=False,
						y1rev=False,
						points=False,xlabel=None,y1label=None,
						fig=None,
						show_mix=False,show_burn=True):
		"""Plots mulitple profiles either given as a list of mod numbers or an index over the history data"""
		if fig==None:
			fig=plt.figure(figsize=(12,12))
		if ax==None:
			ax=fig.add_subplot(111)
		
		if mods is not None:
			cm=[cmap(i) for i in np.linspace(0.0,0.9,len(mods))]
			for i in range(len(mods)):
				model=mods[i]
				self.plotProfile(m,model=model,xaxis=xaxis,show=False,ax=ax,fig=fig,
								xmin=xmin,xmax=xmax,xlog=xlog,xlabel=xlabel,
							xrev=xrev,y1rev=y1rev,points=points,
							y1=y1,y1log=y1log,y1col='k',
							y1label=y1label,show_mix=show_mix,show_burn=show_burn,
							show_mix_line=True,show_burn_line=True)
		elif index is not None:
			cm=[cmap(i) for i in np.linspace(0.0,0.9,np.count_nonzero(index))]
			for i in m.hist.data["model_number"][index]:
				model=m.hist.data["model_number"][index][i]
				self.plotProfile(m,model=model,xaxis=xaxis,show=False,ax=ax,xmin=xmin,
								xmax=xmax,xlog=xlog,xlabel=xlabel,
							xrev=xrev,y1rev=y1rev,points=points,
							y1=y1,y1log=y1log,y1col=cm[i],
							y1label=y1label,fig=fig) 
		
		ax.legend(loc=0,fontsize=12)
		
		if show:
			plt.show()
			
			
	def plotGrid2(self,m,show=True):
		"""Why not grid1? trying to copy mesa's grids and grid2 is easier for now"""
		fig=plt.figure(figsize=(12,12))
		fig.subplots_adjust(wspace=.5)
		fig.subplots_adjust(hspace=.5)
		ax=plt.subplot(2,2,1)
		self.plotTRho(m,ax=ax,show=False)
		
		ax=plt.subplot(2,4,5)
		self.plotHR(m,ax=ax,maxMod=m.prof.head['model_number'],show=False)
		
		ax=plt.subplot(2,4,6)
		self.plotHistory(m,ax=ax,show=False,xaxis='log_center_T',y1='log_center_Rho',y1L='linear',
							minMod=0,maxMod=m.prof.head['model_number'],y1col='k',
							xlabel=self.labels('teff',log=True,center=True),
							y1label=self.labels('rho',log=True,center=True))
		
		ax=plt.subplot(1,2,2)
		self.plotAbun(m,ax=ax,show=False,xlabel=self.labels('mass'))
		
		if show==True:
			plt.show()
			
	def plotSliderProf(self,m,func,*args,**kwargs):
		from matplotlib.widgets import Slider, Button, RadioButtons
		class DiscreteSlider(Slider):
			"""A matplotlib slider widget with discrete steps."""
			def __init__(self, *args, **kwargs):
				"""
				Identical to Slider.__init__, except for the new keyword 'allowed_vals'.
				This keyword specifies the allowed positions of the slider
				"""
				self.allowed_vals = kwargs.pop('allowed_vals',None)
				self.previous_val = kwargs['valinit']
				Slider.__init__(self, *args, **kwargs)
				if self.allowed_vals is None:
					self.allowed_vals = [self.valmin,self.valmax]
		
			def set_val(self, val):
				discrete_val = self.allowed_vals[abs(val-self.allowed_vals).argmin()]
				xy = self.poly.xy
				xy[2] = discrete_val, 1
				xy[3] = discrete_val, 0
				self.poly.xy = xy
				self.valtext.set_text(self.valfmt % discrete_val)
				if self.drawon: 
					self.ax.figure.canvas.draw()
				self.val = discrete_val
				if self.previous_val!=discrete_val:
					self.previous_val = discrete_val
					if not self.eventson: 
						return
					for cid, func in self.observers.items():
						func(discrete_val)

		fig=plt.figure(figsize=(12,12))
		ax=plt.axes([0.1,0.15,0.7,0.75])
		f=getattr(self,func)
		
		m.loadProfile(num=1)
		f(m,fig=fig,ax=ax,show=False,show_title_model=True,*args,**kwargs)
		
		axcolor = 'white'
		axmodels = plt.axes([0.15, 0.025, 0.75, 0.03], axisbg=axcolor,label='slider')
		
		num_models=np.size(m.prof_ind['model'])
		
		smodels=DiscreteSlider(axmodels,'Model',0.0,1.0*num_models-1.0,valinit=0.0,allowed_vals=np.arange(0.0,num_models,1.0))
		
		smodels.set_val(np.argmin(np.abs(m.prof_ind['model']-m.prof.model_number)))
		
		smodels.ax.set_xticks(np.arange(0.0,num_models,1.0))
		smodels.ax.set_xticklabels([])
		smodels.valtext.set_visible(False)
		
		def update(val):
			mo = smodels.val
			m.loadProfile(num=m.prof_ind['model'][int(mo)])
			xmin,xmax=ax.get_xlim()
			ymin,ymax=ax.get_ylim()
			ymin2=None
			ymax2=None
			for i in fig.axes:
				if '_ax2' in i.get_label():
					fig.delaxes(i)
					ymin2,ymax2=i.get_ylim()
			plt.sca(ax)
			plt.cla()
			try:
				kwargs.pop('xmax')
			except KeyError:
				pass
			try:
				kwargs.pop('xmin')
			except KeyError:
				pass
			try:
				kwargs.pop('yrng')
			except KeyError:
				pass
			f(m,fig=fig,ax=ax,xmin=xmin,xmax=xmax,yrng=[ymin,ymax],y2rng=[ymin2,ymax2],show_title_model=True,show=False,*args,**kwargs)
			#fig.canvas.draw_idle()
			fig.canvas.draw()
		
		smodels.on_changed(update)
		
		plt.show()
			
class debug_logs(object):
	def __init__(self,folder):
		self.folder=folder
		self.names=[]
		self.size=[]
		self.data=[]	

	def _load_log_size(self):
		with open(os.path.join(self.folder,'size.data')) as f:
			for line in f:
				self.size=[int(i) for i in line.strip('\n').split()]
				break
		
	def _load_log_names(self):
		with open(os.path.join(self.folder,'names.data')) as f:
			for line in f:
				self.names.append(line.strip(' \n'))
	
	def load_all_logs(self):		
		self._load_log_names()
		self._load_log_size()
		
		for i in self.names:
			self.data.append([])
			x=np.genfromtxt(os.path.join(self.folder,i+'.log'))
			self.data[-1]=x.reshape(self.size[::-1])
	
	def plot_log(self,name='',save=False,folder=None,iter_min=-1,iter_max=99999999,zone_min=-1,zone_max=99999999,log=False):

		idx=self.names.index(name)

		if np.all(self.data[idx]==0.0):
			print('Empty ',name)
			return
		
		if folder is not None:
			self.folder=folder
		
		plt.figure(figsize=(12,12))
		plt.title(name.replace('_',' '))

		
		shp=np.shape(self.data[idx])
		
		iter_min=max(iter_min,0)
		iter_max=min(iter_max,shp[0])
		zone_min=max(zone_min,0)
		zone_max=min(zone_max,shp[1])

		vmin=np.nanmin(self.data[idx][iter_min:iter_max,zone_min:zone_max])
		vmax=np.nanmax(self.data[idx][iter_min:iter_max,zone_min:zone_max])

		if vmin>=0.0:
			vmin=0.0
			cmap='Reds'
		else:
			cmap='seismic'
			if np.abs(vmin) < np.abs(vmax):
				vmin=-vmax
			else:
				vmax=np.abs(vmin)
				
		d=self.data[idx][iter_min:iter_max,zone_min:zone_max]
		if log:
			d=np.log10(np.abs(d))
			vmin=np.nanmin(d)
			vmax=np.nanmax(d)

		plt.imshow(d,extent=(zone_min,zone_max,iter_min,iter_max),aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax,
			 interpolation='nearest',origin='lower')

		plt.xlim(zone_max,zone_min)
		plt.ylim(iter_min,iter_max)
		plt.xlabel('Zone')
		plt.ylabel('Iter')
		cb=plt.colorbar()
		cb.solids.set_edgecolor("face")
		if log:
			cb.set_label('log abs')
		
		if save:
			plt.savefig(os.path.join(self.folder,name+'.pdf'))
		else:
			plt.show()
		
		plt.close()
		
	
	def plot_all_logs(self,iter_min=-1,iter_max=99999999,zone_min=-1,zone_max=99999999,folder=None):
		if folder is not None:
			self.folder=folder
		for i in self.names:
			self.plot_log(name=i,save=True,folder=self.folder,iter_min=iter_min,iter_max=iter_max,zone_min=zone_min,zone_max=zone_max)
			print("Done ",i)

	def summary(self,iter_min=-1,iter_max=99999999,zone_min=-1,zone_max=99999999):
		print("Name Max Min Mean")
		for i in range(len(self.names)):
			if ("corr_" in self.names[i] or "equ_" in self.names[i]) and "delta_" not in self.names[i]:
				shp=np.shape(self.data[i])
				iter_min=max(iter_min,0)
				iter_max=min(iter_max,shp[0])
				zone_min=max(zone_min,0)
				zone_max=min(zone_max,shp[1])
				print(self.names[i],end=' ')
				print(np.log10(np.abs(np.nanmax(self.data[i][iter_min:iter_max,zone_min:zone_max]))),end=' ')
				print(np.log10(np.abs(np.nanmin(self.data[i][iter_min:iter_max,zone_min:zone_max]))),end=' ')
				print(np.log10(np.abs(np.nanmean(self.data[i][iter_min:iter_max,zone_min:zone_max]))))

	
class debug(object):
	def __init__(self,folder='plot_data'):
		self.solve=debug_logs(os.path.join(folder,'solve_logs'))
		self.res=debug_logs(os.path.join(folder,'residual_logs'))

		self.jacobian_default=folder

	def load_res(self):
		self.res.load_all_logs()

	def load_solve(self):
		self.solve.load_all_logs()

	def plot_res(self,iter_min=-1,iter_max=99999999,zone_min=-1,zone_max=99999999):
		self.res.plot_all_logs(iter_min=iter_min,iter_max=iter_max,zone_min=zone_min,zone_max=zone_max)
		
	def plot_solve(self,iter_min=-1,iter_max=99999999,zone_min=-1,zone_max=99999999):
		self.solve.plot_all_logs(iter_min=iter_min,iter_max=iter_max,zone_min=zone_min,zone_max=zone_max)
	
###################################
	
	def _load_jacobian_size(self,folder):
		cols=0
		rows=0
		with open(os.path.join(folder,'jacobian_cols.data')) as f:
			for i,l in enumerate(f):
				pass
			cols=i+1
		with open(os.path.join(folder,'jacobian_rows.data')) as f:
			for i,l in enumerate(f):
				pass
			rows=i+1
		return cols,rows
		
	def _load_jacobian_names(self,folder):
		names=[]
		with open(os.path.join(folder,'names.data')) as f:
			for line in f:
				names.append(line.strip(' \n'))
		return names
				
	def _get_index(self,jacob,name):
		idx=None
		jdx=None
		try:
			idx=self.jacobian_folds.index(jacob)
			jdx=self.jacobian_names[idx].index(name)
		except:
			print("call load_all_jacobian")
			raise
		return (idx,jdx)
	
	def _load_all_jacobian_names(self,folder):
		self.jacob_type=["jacobian_data","jacobian_diff_data","jacobian_rel_diff_data","numerical_jacobian"]
		self.jacobian_names=[]
		self.jacobian_folds=[]
		self.jacobian_size=[]
		for i in self.jacob_type:
			try:
				self.jacobian_names.append(self._load_jacobian_names(os.path.join(folder,i)))
				self.jacobian_folds.append(i)
				print("Reading ",i)
				c,r=self._load_jacobian_size(os.path.join(folder,i))
				self.jacobian_size.append([r,c])
			except:
				pass
	
	def load_all_all_jacobian(self,folder=None):
		if folder is None:
			folder=self.jacobian_default
		
		self._load_all_jacobian_names(folder)
		
		self.jacobian_data=[]
		for idx,i in enumerate(self.jacobian_folds):
			self.jacobian_data.append([])
			for jdx,j in enumerate(self.jacobian_names[idx]):
				print("Reading ",i,j)
				x=np.genfromtxt(os.path.join(folder,i,j+'.data'))
				self.jacobian_data[-1].append(x)
	
	def load_all_jacobian(self,jacob='jacobian_data',folder=None):
		if folder is None:
			folder=self.jacobian_default
		
		self._load_all_jacobian_names(folder)
		
		self.jacobian_data=[[],[],[],[],[]]
		idx=self.jacobian_folds.index(jacob)
		for jdx,j in enumerate(self.jacobian_names[idx]):
			print("Reading ",jacob,j)
			x=np.genfromtxt(os.path.join(folder,jacob,j+'.data'))
			self.jacobian_data[idx].append(x)
	
	def list_avail_jacobians(self):
		try:
			return self.jacobian_folds
		except:
			print("call load_all_jacobian")
			raise
			
	def list_avail_names(self,name):
		try:
			idx=self.jacobian_folds.index(name)
			return self.jacobian_names[idx]
		except:
			print("call load_all_jacobian")
			raise
	
	def get_jacob_data(self,jacob,name):
		idx,jdx=self._get_index(jacob,name)
		return self.jacobian_data[idx][jdx]
	
	def plot_all_all_jacob_data(self,folder=None):
		if folder is not None:
			folder=self.jacobian_default

		for j in self.jacob_type:
			for i in iter(self.list_avail_names(j)):
				self.plot_jacob_data(jacob=j,name=i,save=True,folder=folder)
				print("Plotting ",j,i)
				
	def plot_all_jacob_data(self,jacob='jacobian_data',folder=None):
		if folder is not None:
			folder=self.jacobian_default

		for i in iter(self.list_avail_names(jacob)):
			self.plot_jacob_data(jacob=jacob,name=i,save=True,folder=folder)
			print("Plotting ",jacob,i)

	def plot_jacob_data(self,jacob,name,save=False,folder=None,iter_min=-1,iter_max=99999999,zone_min=-1,zone_max=99999999):
		if folder is None:
			folder=self.jacobian_default
		
		idx,jdx=self._get_index(jacob,name)
		data=self.get_jacob_data(jacob,name)
		
		if np.all(data[idx]==0.0):
			print('Empty ',jacob,name)
			return

		plt.figure(figsize=(12,12))
		plt.title(name.replace('_',' '))
		
		shp=np.shape(data)
		zone_min=max(zone_min,0)
		zone_max=min(zone_max,shp[1])

		vmin=np.nanmin(data[:,zone_min:zone_max])
		vmax=np.nanmax(data[:,zone_min:zone_max])
		
		if np.abs(vmin) < np.abs(vmax):
			vmin=-vmax
		else:
			vmax=np.abs(vmin)

		plt.imshow(data[:,zone_min:zone_max],extent=(zone_min,zone_max,-1,1),aspect='auto',cmap='seismic',vmin=vmin,vmax=vmax,
			 interpolation='nearest',origin='lower')

		plt.xlim(zone_min,zone_max)
		plt.yticks([-1,0.0,1.0],['mm','00','pm'])
		plt.xlabel('Zone')
		plt.ylabel('Iter')
		cb=plt.colorbar()
		cb.solids.set_edgecolor("face")
		
		if save:
			plt.savefig(os.path.join(folder,jacob,name+'.pdf'))
		else:
			plt.show()
		
		plt.close()

	def find_bad_data(self,jacob):
		for name in iter(self.list_avail_names(jacob)):
			x=self.get_jacob_data(jacob,name)
			print(name,np.maximum(np.abs(np.nanmax(x[1])),np.abs(np.nanmin(x[1]))),np.nanmax(x[1]),np.nanmin(x[1]),np.nanargmax(x[1]),np.nanargmin(x[1]))
	

class plotEOS(object):
   
	def __init__(self):
		self.data=[]
		self.name=[]
		self.teff=0
		self.rho=0

	def load(self,folder='plot_data_DT'):
		self.folder=folder
		self.teff=np.genfromtxt(folder+'/logT.data')
		self.rho=np.genfromtxt(folder+'/logRho.data')
		
		for i in glob.glob(folder+'/*.data'):
			if '/logT.data' not in i and '/logRho.data' not in i and '/params.data' not in i:
				print('Load',i)
				self.name.append(i.replace(folder,'').replace('.data','').replace('/',''))
				x=np.genfromtxt(i)
				self.data.append(x.reshape((np.size(self.rho),np.size(self.teff))))
		
	def plot(self,name,save=False,only_neg=False):
		idx=self.name.index(name)

		if np.all(self.data[idx]==0.0):
			print('Empty ',name)
			return
		
		plt.figure(figsize=(12,12))
		plt.title(name.replace('_',' '))

		data=self.data[idx]
	
		vmin=np.nanmin(data)
		vmax=np.nanmax(data)

		if vmin>=0.0:
			vmin=0.0
			cmap='Reds'
		else:
			cmap='seismic'
			if np.abs(vmin) < np.abs(vmax):
				vmin=-vmax
			else:
				vmax=np.abs(vmin)
				
		if only_neg:
			vmax=0.0
			vmin=np.nanmin(data)
			cmap='Reds_r'
			data[data>0.0]=np.nan
			

		plt.imshow(data,extent=(np.nanmin(self.rho),np.nanmax(self.rho),np.nanmin(self.teff),np.nanmax(self.teff)),aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax,
				interpolation='nearest',origin='lower')

		plt.xlim(np.nanmin(self.rho),np.nanmax(self.rho))
		plt.ylim(np.nanmin(self.teff),np.nanmax(self.teff))
		plt.xlabel('LogRho')
		plt.ylabel('LogT')
		cb=plt.colorbar()
		cb.solids.set_edgecolor("face")
		
		if save:
			plt.savefig(os.path.join(self.folder,name+'.pdf'))
		else:
			plt.show()
		
		plt.close()
		


	def plot_all(self,only_neg=False):
		for i in self.name:
			print('Plot',i)
			self.plot(name=i,save=True,only_neg=only_neg)
	

class plotNet(object):
	def __init__(self):
		self.data=[]
		self.name=[]
		
	def load(self,filename):
		from io import BytesIO
		d=subprocess.check_output(os.path.expandvars('$MESA_DIR')+"/rates/test/show_rates "+filename,shell=True)
		
		self.name.append(os.path.basename(filename))
		if type(d) is not type('a'):
			data=d.decode().replace('D','E')
		
		data=data.encode()
		data=np.genfromtxt(BytesIO(data),names=['t8','rate'],skip_header=4)
		
		self.data.append(data)
	
	def load_all(self,folder='cache'):
		for i in glob.glob(os.path.expandvars('$MESA_DIR')+'/data/rates_data/'+folder+'/r_*.bin'):
			print(i)
			self.load(i)
	
	def plot(self,name,show=True,trng=None,rrng=None):
		
		for n,data in zip(self.name,self.data):
			if n==name:
				d=data
				
		fig=plt.figure(figsize=(12,12))
		ax=fig.add_subplot(111)
		ax.plot(np.log10(d['t8']*10**8),np.log10(d['rate']),linewidth=2)
		ax.scatter(np.log10(d['t8']*10**8),np.log10(d['rate']))
		ax.set_xlabel(r'$\rm{T}_8$')
		ax.set_ylabel(r'$\rm{Rate}$')
		ax.set_title(name.replace('_','\_'))
		if trng is not None:
			ax.set_xlim(trng)
		if rrng is not None:
			ax.set_ylim(rrng)
			
		ax.autoscale()
		if show:
			plt.show()
		else:
			plt.savefig(name.replace('.bin','')+'.pdf')
		plt.close(fig)
		
	def plot_all(self,show=False,trng=None,rrng=None):
		for i in self.name:
			print(i)
			self.plot(i,show,trng,rrng)
			
			

			
class debug_mesh(object):
	def __init__(self):
		self.data_old=[]
		self.data_new=[]

	def load(self,folder='mesh_plot_data',mesh_new='new_mesh.data',mesh_old='mesh_plan.data'):
		self.data_old=np.genfromtxt(os.path.join(folder,mesh_old),names=True)
		self.data_new=np.genfromtxt(os.path.join(folder,mesh_new),names=True)

	def plot_gval(self,data,show=True):
		for i in data.dtype.names:
			if i.startswith('gval_'):
				plt.plot(data['mass'],data[i],label=data.replace('_','\_'),linewidth=2)
	
		plt.legend(loc=0)	
		if show:
			plt.show()

		
		
class plotNeu(object):
   
	def __init__(self):
		self.data=[]
		self.name=[]
		self.teff=0
		self.rho=0

	def load(self,folder='plot_data'):
		self.folder=folder
		self.teff=np.genfromtxt(folder+'/tmp.data')
		self.rho=np.genfromtxt(folder+'/rho.data')
		
		for i in glob.glob(folder+'/*.data'):
			if '/tmp.data' not in i and '/rho.data' not in i:
				print('Load',i)
				self.name.append(i.replace(folder,'').replace('.data','').replace('/',''))
				x=np.genfromtxt(i)
				self.data.append(x.reshape((np.size(self.rho),np.size(self.teff))))
		
	def plot(self,name,save=False,only_neg=False):
		idx=self.name.index(name)

		if np.all(self.data[idx]==0.0):
			print('Empty ',name)
			return
		
		plt.figure(figsize=(12,12))
		plt.title(name.replace('_',' '))

		data=self.data[idx]
		
		vmin=np.nanmin(data)
		vmax=np.nanmax(data)
		
		label=''
		if vmax>100.0 or vmin<-100.0:
			data=np.log10(np.abs(data))
			label='Log'
			vmin=np.nanmin(data)
			vmax=np.nanmax(data)

		if vmin>=0.0:
			vmin=0.0
			cmap='Reds'
		else:
			cmap='seismic'
			if np.abs(vmin) < np.abs(vmax):
				vmin=-vmax
			else:
				vmax=np.abs(vmin)
				
		if only_neg:
			vmax=0.0
			vmin=np.nanmin(data)
			cmap='Reds_r'
			data[data>0.0]=np.nan
			

		plt.imshow(data,extent=(np.nanmin(self.rho),np.nanmax(self.rho),np.nanmin(self.teff),np.nanmax(self.teff)),aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax,
				interpolation='nearest',origin='lower')

		plt.xlim(np.nanmin(self.rho),np.nanmax(self.rho))
		plt.ylim(np.nanmin(self.teff),np.nanmax(self.teff))
		plt.xlabel('LogRho')
		plt.ylabel('LogT')
		cb=plt.colorbar()
		cb.solids.set_edgecolor("face")
		cb.set_label(label)
		
		if save:
			plt.savefig(os.path.join(self.folder,name+'.pdf'))
		else:
			plt.show()
		
		plt.close()
		


	def plot_all(self,only_neg=False):
		for i in self.name:
			print('Plot',i)
			self.plot(name=i,save=True,only_neg=only_neg)


