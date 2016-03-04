#Copyright (c) 2015, Robert Farmer rjfarmer@asu.edu

#This program is free software; you can redistribute it and/or
#modify it under the terms of the GNU General Public License
#as published by the Free Software Foundation; either version 2
#of the License, or (at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program; if not, write to the Free Software
#Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


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

def randf(N,a,b):
	return a + (b - a) * (np.random.random_integers(N) - 1) / (N - 1.)

class data(object):
	def __init__(self):
		self.data={}
		self.head={}
		

	def __getattr__(self, name):
		x=None
		
		try:
			x=self.data[name]
		except:
			try:
				x=np.atleast_1d(self.head[name])[0]
			except:
				raise NameError
		if x is not None:
			return x
		else:
			raise NameError
	
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
			raise NameError

	def loadFile(self,filename):
		numLines=self._filelines(filename)
		self.head=np.genfromtxt(filename,skip_header=1,skip_footer=numLines-4,names=True)
		self.data=np.genfromtxt(filename,skip_header=5,names=True)
		self.head_names=self.head.dtype.names
		self.data_names=self.data.dtype.names

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
	
	def loadHistory(self,f="",filename_in=None):
		"""
		Reads a MESA history file.
		
		Optional:
		f: Folder in which history.data exists, if not present uses self.log_fold, if thats
		not set trys the folder LOGS/
		filename_in: Reads the file given by name
		
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

		self.hist.loadFile(filename)

		#Inspired by http://www.mesastar.org/tools-utilities/python-based-stuff/history-log-scrubber/view
		#to remove bad lines
		while (np.any(np.diff(self.hist.data["model_number"])<=0.0)):
			rev=np.copy(self.hist.data["model_number"][::-1])
			self.hist.data=self.hist.data[np.concatenate(([True],np.diff(rev)<0))[::-1]]
		
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
	
		
	def loadProfile(self,f='',num=None,prof=-1,mode='nearest'):
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
					  self.colors['clr_Beige'], #Thermohaline
					  self.colors['clr_BrightBlue'], #Rotation
					  self.colors['clr_Coral'], #Minimum
					  self.colors['clr_Tan'] #Anonymous
					  ]
					
	
	def _getMESAPath(self):
		self.mesa_dir=os.getenv("MESA_DIR")
		if self.mesa_dir==None:
			raise ValueError("Must set $MESA_DIR in terminal")
			
	def _loadBurnData(self):
		self._getMESAPath()
		dataDir=self.mesa_dir+"/data/star_data/plot_info/"
		
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
		if 'lum' in label:
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
			l=None
			
		return l
		
	def safeLabel(self,label,axis):
		outLabel=''
		if label is not None:
			outLabel=label
		else:
			outLabel=self.labels(axis)
			if outLabel is not None:
				outLabel=outLabel
			else:
				outLabel=axis.replace('_',' ')
		return outLabel
		
	def _listAbun(self,data):
		abun_list=[]
		names=data.data_names
		for j in names:
			if "log_" in j:
				i=j[4:]
				l="log_"
			else:
				i=j
				l=""
			if len(i)<=5 and len(i)>=2:
				if i[0].isalpha() and (i[1].isalpha() or i[1].isdigit()) and any(char.isdigit() for char in i) and i[-1].isdigit():
					if (len(i)==5 and i[-1].isdigit() and i[-2].isdigit()) or len(i)<5:
						abun_list.append(l+i)
						lout=l
		return abun_list,lout

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


	def _setMixRegionsCol(self,kip=False,mix=False):		
		cmap = mpl.colors.ListedColormap(self.mix_col)
	
		cmap.set_over((1., 1., 1.))
		cmap.set_under((0., 0., 0.))
		if mix:
			bounds=[0.0,1.01,2.01,3.01,4.01,5.01,6.01,7.01,8.01,9.01]
		if kip:
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
		ylim=ax.get_ylim()
		
		if show_x:
			yy=np.zeros(np.size(x))
			if yrng is not None:
				yy[:]=yrng[0]
			else:
				yy[:]=ylim[0]
			size=90

		yy=y
		size=60
	
		cmap,norm=self._setMixRegionsCol(mix=True)
		
		isSet=None
		for mixLabel in ['mixing_type','conv_mixing_type','mlt_mixing_type']:
			try:
				col=m.prof.data[mixLabel]
				isSet=True
				break
			except:
				continue
			
		if isSet is None:
			raise(ValueError,"Need mixing type in profile file for showing mix regions, either its mixing_type or conv_mixing_type")
		
		if ind is not None:
			col=col[ind]
		
		ax.scatter(x,yy,c=col,s=size,cmap=cmap,norm=norm,linewidths=0)
	
		ax.set_ylim(ylim)
	
	def _annotateLine(self,ax,x,y,num_labels,xmin,xmax,text,line,fontsize=mpl.rcParams['font.size']-12):
		ind=np.argsort(x)
		xx=x[ind]
		yy=y[ind]
		for ii in range(1,num_labels+1):
			ind=(xx>=xmin)&(xx<=xmax)
			f = interpolate.interp1d(xx[ind],yy[ind])
			xp1=((xmax-xmin)*(ii/(num_labels+1.0)))+xmin
			yp1=f(xp1)
			ax.annotate(text, xy=(xp1,yp1), xytext=(xp1,yp1),color=line.get_color(),fontsize=fontsize)
	
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
			ax.plot([0,0],[0,0],c='w',label=i,alpha=0.0)
				
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
				ax.plot([x[ind][0],x[ind][0]],[np.nanmin(y),np.nanmax(y)],'--',color=self.colors[j],linewidth=2)
				out.append(x[ind][0])
				outc.append(l)
		
		ax2=ax.twiny()
		ax2.plot(ax.get_xlim(),ax.get_ylim())
		ax2.cla()
		ax2.set_xlim(ax.get_xlim())
		ax2.set_xticks(out)
		ax2.set_xticklabels(outc)
		
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
				axis.plot(0,0,c='w')
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
				axis.plot(0,0,c='w')
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
			

	def _plotAnnotatedLine(self,ax,x,y,fy,xmin,xmax,ymin=None,ymax=None,annotate_line=False,label='',
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
				line, =ax.plot(x,y,linestyle=linestyle,c=linecol,linewidth=linewidth)
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
	
	def plotAbun(self,m,model=None,show=True,ax=None,xaxis='mass',xmin=None,xmax=None,yrng=[-3.0,1.0],
					cmap=plt.cm.gist_ncar,num_labels=3,xlabel=None,points=False,abun=None,abun_random=False,
				show_burn=False,show_mix=False,fig=None,fx=None,fy=None,modFile=False,
				show_title_name=False,show_title_model=False,show_title_age=False,annotate_line=True,linestyle='-',
				colors=None):
		if fig==None:
			fig=plt.figure()
		if ax==None:
			ax=fig.add_subplot(111)
		#m.loadProfile(num=int(model))
		
		if model is not None:
			try:
				if m.prof.head["model_number"]!=model:
					m.loadProfile(num=int(model))
			except:
				m.loadProfile(num=int(model))
			
		if modFile:
			x,xrngL,mInd=self._setXAxis(np.cumsum(m.mod_dat["dq"][::-1])*m._fds2f(m.mod_head[1]),xmin,xmax,fx)
		else:
			x,xrngL,mInd=self._setXAxis(m.prof.data[xaxis],xmin,xmax,fx)

		if abun is None:
			abun_list,log=self._listAbun(m.prof)
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

		if colors is None:
			plt.gca().set_color_cycle([cmap(i) for i in np.linspace(0.0,0.9,num_plots)])
		else:
			plt.gca().set_color_cycle(colors)
		
		
		for i in abun_list:
			self._plotAnnotatedLine(ax=ax,x=x,y=m.prof.data[i],fy=fy,xmin=xrngL[0],xmax=xrngL[1],
									ymin=yrng[0],ymax=yrng[1],annotate_line=annotate_line,
									label=self.safeLabel(None,i),points=points,ylog=abun_log,num_labels=num_labels,linestyle=linestyle)
			
		if show_burn:
			self._plotBurnRegions(m,ax,x,m.prof.mass,show_line=False,show_x=True,yrng=yrng,ind=mInd)

		if show_mix:
			self._plotMixRegions(m,ax,x,m.prof.mass,show_line=False,show_x=True,yrng=yrng,ind=mInd)
			
			
		ax.set_xlabel(self.safeLabel(xlabel,xaxis))
		ax.set_ylabel(r'$\log_{10}$ Abundance')
		
		if show_title_name or show_title_model or show_title_age:
			self.setTitle(ax,show_title_name,show_title_model,show_title_age,'Abundances',m.prof.head["model_number"],m.prof.head["star_age"])
		
		
		if show:
			plt.show()
			

	def plotDynamo(self,m,xaxis='mass',model=None,show=True,ax=None,xmin=None,xmax=None,xlabel=None,y1rng=None,y2rng=None,
					show_burn=False,show_mix=False,legend=True,annotate_line=True,fig=None,fx=None,fy=None,
				show_title_name=False,show_title_model=False,show_title_age=False,show_rotation=True):
		if fig==None:
			fig=plt.figure()
			
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
		
		
		ax1_1.plot(0,0,c='w')
		ax1_2.plot(0,0,c='w')
		ax2_1.plot(0,0,c='w')
		ax2_2.plot(0,0,c='w')
		
		if ax==None:
			ax=fig.add_subplot(111)
	
		ax2=ax.twinx()
	
		if model is not None:
			try:
				if m.prof.head["model_number"]!=model:
					m.loadProfile(num=int(model))
			except:
				m.loadProfile(num=int(model))
			
		x,xrngL,mInd=self._setXAxis(m.prof.data[xaxis],xmin,xmax,fx)
		
		#ind=(m.prof.data['dynamo_log_B_r']>-90)
		ax.plot(m.prof.data[xaxis],m.prof.data['dynamo_log_B_r'],label=r'$B_r$',linewidth=2,c='g')
		#ind=mInd&(m.prof.data['dynamo_log_B_phi']>-90)
		ax.plot(m.prof.data[xaxis],m.prof.data['dynamo_log_B_phi'],label=r'$B_{\phi}$',linewidth=2,c='b')
		
		if show_rotation:
			ax2.plot(m.prof.data[xaxis],np.log10(m.prof.data['omega']),'--',label=r'$\log_{10} \omega$',linewidth=2,c='r')
		#ind=mInd&(m.prof.data['dynamo_log_B_phi']>-90)
			ax2.plot(m.prof.data[xaxis],np.log10(m.prof.data['j_rot'])-20.0,'--',label=r'$\log_{10} j [10^{20}]$',linewidth=2,c='k')


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

		ax.set_xlabel(self.safeLabel(xlabel,xaxis))
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
				show_title_name=False,show_title_model=False,show_title_age=False,points=False,show_core=False):
		if fig==None:
			fig=plt.figure()
		if ax==None:
			ax=fig.add_subplot(111)
			
		if model is not None:
			try:
				if m.prof.head["model_number"]!=model:
					m.loadProfile(num=int(model))
			except:
				m.loadProfile(num=int(model))
			
		x,xrngL,mInd=self._setXAxis(m.prof.data[xaxis],xmin,xmax,fx)

		for i in m.prof.data_names:         
			if "am_log_D" in i:
				px,py=self._plotAnnotatedLine(ax=ax,x=x,y=m.prof.data[i],fy=fy,xmin=xrngL[0],xmax=xrngL[1],
										ymin=yrng[0],ymax=yrng[1],annotate_line=annotate_line,
										label=r"$D_{"+i.split('_')[3]+"}$",points=points,
										ylog=True,num_labels=num_labels)

		if show_burn:
			self._plotBurnRegions(m,ax,px,m.prof.data[i],show_line=False,show_x=True,ind=mInd)

		if show_mix:
			self._plotMixRegions(m,ax,px,m.prof.data[i],show_line=False,show_x=True,ind=mInd)
			
		if show_core:
			self._showMassLocHist(m,fig,ax,x,y,mInd)

		if legend:
			ax.legend(loc=0)

		ax.set_xlabel(self.safeLabel(xlabel,xaxis))
		self.setTitle(ax,show_title_name,show_title_model,show_title_age,'Ang mom',m.prof.head["model_number"],m.prof.head["star_age"])
		
		
		if show:
			plt.show()
			
	def plotBurn(self,m,xaxis='mass',model=None,show=True,ax=None,xmin=None,xmax=None,xlabel=None,
				cmap=plt.cm.gist_ncar,yrng=[0.0,10.0],num_labels=7,burn_random=False,points=False,
				show_burn=False,show_mix=False,fig=None,fx=None,fy=None,
				show_title_name=False,show_title_model=False,show_title_age=False,annotate_line=True):
		if fig==None:
			fig=plt.figure()
		if ax==None:
			ax=fig.add_subplot(111)
			
		if model is not None:
			try:
				if m.prof.head["model_number"]!=model:
					m.loadProfile(num=int(model))
			except:
				m.loadProfile(num=int(model))
			
		x,xrngL,mInd=self._setXAxis(m.prof.data[xaxis],xmin,xmax,fx)


		burn_list=self._listBurn(m.prof)
		num_plots=len(burn_list)
		
		if burn_random:
			random.shuffle(burn_list)
				
		plt.gca().set_color_cycle([cmap(i) for i in np.linspace(0.0,0.9,num_plots)])
			
		for i in burn_list:
			px,py=self._plotAnnotatedLine(ax=ax,x=x,y=m.prof.data[i],fy=fy,xmin=xrngL[0],
									xmax=xrngL[1],ymin=yrng[0],ymax=yrng[1],annotate_line=annotate_line,
									label=self.safeLabel(None,i),points=points,ylog=True,num_labels=num_labels)

		
		if show_burn:
			self._plotBurnRegions(m,ax,px,py,show_line=False,show_x=True,ind=mInd)

		if show_mix:
			self._plotMixRegions(m,ax,px,py,show_line=False,show_x=True,ind=mInd)
		
		ax.set_xlabel(self.safeLabel(xlabel,xaxis))
		self.setTitle(ax,show_title_name,show_title_model,show_title_age,'Burn',m.prof.head["model_number"],m.prof.head["star_age"])
		
		
		if show:
			plt.show()
			
	def plotMix(self,m,xaxis='mass',model=None,show=True,ax=None,xmin=None,xmax=None,xlabel=None,
				cmap=plt.cm.gist_ncar,yrng=[0.0,5.0],num_labels=7,mix_random=False,points=False,
				show_burn=False,fig=None,fx=None,fy=None,
				show_title_name=False,show_title_model=False,show_title_age=False,annotate_line=True):
		if fig==None:
			fig=plt.figure()
		if ax==None:
			ax=fig.add_subplot(111)
			
		if model is not None:
			try:
				if m.prof.head["model_number"]!=model:
					m.loadProfile(num=int(model))
			except:
				m.loadProfile(num=int(model))
			
		x,xrngL,mInd=self._setXAxis(m.prof.data[xaxis],xmin,xmax,fx)

		mix_list=self._listMix(m.prof)
		num_plots=len(mix_list)
		
		if mix_random:
			random.shuffle(mix_list)
				
		plt.gca().set_color_cycle([cmap(i) for i in np.linspace(0.0,0.9,num_plots)])
			
		for i in mix_list:
			px,py=self._plotAnnotatedLine(ax=ax,x=x,y=m.prof.data[i],fy=fy,xmin=xrngL[0],
									xmax=xrngL[1],ymin=yrng[0],ymax=yrng[1],
									annotate_line=annotate_line,label=i.split('_')[2],
									points=points,ylog=False,num_labels=num_labels)
		
		if show_burn:
			self._plotBurnRegions(m,ax,px,py,show_line=False,show_x=True,ind=mInd)
			
		
		ax.set_xlabel(self.safeLabel(xlabel,xaxis))
		self.setTitle(ax,show_title_name,show_title_model,show_title_age,'Mixing',m.prof.head["model_number"],m.prof.head["star_age"])
		
		
		if show:
			plt.show()

	def plotBurnSummary(self,m,xaxis='model_number',minMod=0,maxMod=-1,show=True,ax=None,xmin=None,xmax=None,xlabel=None,
				cmap=plt.cm.nipy_spectral,yrng=[0.0,10.0],num_labels=7,burn_random=False,points=False,
				show_burn=False,show_mix=False,fig=None,fx=None,fy=None,annotate_line=True,show_core=False):
		if fig==None:
			fig=plt.figure()
		if ax==None:
			ax=fig.add_subplot(111)
			
		if maxMod<0:
			maxMod=m.hist.data["model_number"][-1]
		modelIndex=(m.hist.data["model_number"]>=minMod)&(m.hist.data["model_number"]<=maxMod)
		
		x,xrngL,mInd=self._setXAxis(m.hist.data[xaxis][modelIndex],xmin,xmax,fx)

			
		burn_list=self._listBurnHistory(m.prof)
		num_plots=len(burn_list)
		
		if burn_random:
			random.shuffle(burn_list)
				
		plt.gca().set_color_cycle([cmap(i) for i in np.linspace(0.0,0.9,num_plots)])
			
		for i in burn_list:
			self._plotAnnotatedLine(ax=ax,x=x,y=m.prof.data[i],fy=fy,xmin=xrngL[0],
									xmax=xrngL[1],ymin=yrng[0],ymax=yrng[1],
									annotate_line=annotate_line,label=self.safeLabel(None,i),
									points=points,ylog=True,num_labels=num_labels)

		if show_burn:
			self._plotBurnRegions(m,ax,x[mInd],y,show_line=False,show_x=True,ind=mInd)

		if show_mix:
			self._plotMixRegions(m,ax,x[mInd],y,show_line=False,show_x=True,ind=mInd)
			
		if show_core:
			self._showMassLocHist(m,fig,ax,x,y,mInd)
		
		ax.set_xlabel(self.safeLabel(xlabel,xaxis))
		ax.set_ylabel(self.labels('log_lum'))

		if show:
			plt.show()

	def plotAbunSummary(self,m,xaxis='model_number',minMod=0,maxMod=-1,show=True,ax=None,xmin=None,xmax=None,xlabel=None,
				cmap=plt.cm.nipy_spectral,yrng=[0.0,10.0],num_labels=7,abun_random=False,points=False,
				show_burn=False,show_mix=False,abun=None,fig=None,fx=None,fy=None,annotate_line=True,linestyle='-',colors=None,
				show_core=False):
		if fig==None:
			fig=plt.figure()
		if ax==None:
			ax=fig.add_subplot(111)
			
		if maxMod<0:
			maxMod=m.hist.data["model_number"][-1]
		modelIndex=(m.hist.data["model_number"]>=minMod)&(m.hist.data["model_number"]<=maxMod)
		
		x,xrngL,mInd=self._setXAxis(m.hist.data[xaxis][modelIndex],xmin,xmax,fx)

			
		if abun is None:
			abun_list,log=self._listAbun(m.hist)
		else:
			abun_list=abun
			
		num_plots=len(abun_list)
		#Helps when we have many elements not on the plot that stretch the colormap
		if abun_random:
			random.shuffle(abun_list)
		
		if colors is None:
			plt.gca().set_color_cycle([cmap(i) for i in np.linspace(0.0,0.9,num_plots)])
		else:
			plt.gca().set_color_cycle([colors(i) for i in np.linspace(0.0,0.9,num_plots)])
			
		for i in abun_list:
			y=m.hist.data["log_total_mass_"+i][mInd]
			self._plotAnnotatedLine(ax=ax,x=x,y=y,fy=fy,xmin=xrngL[0],
									xmax=xrngL[1],ymin=yrng[0],ymax=yrng[1],
									annotate_line=annotate_line,label=self.safeLabel(None,i),
									points=points,ylog=True,num_labels=num_labels,linestyle=linestyle)


		if show_burn:
			self._plotBurnRegions(m,ax,x[mInd],y,show_line=False,show_x=True,ind=mInd)

		if show_mix:
			self._plotMixRegions(m,ax,x[mInd],y,show_line=False,show_x=True,ind=mInd)
			
		if show_core:
			self._showMassLocHist(m,fig,ax,x,y,mInd)
		
		ax.set_xlabel(self.safeLabel(xlabel,xaxis))
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
						y1linelabel=None,show_core_loc=False):
		if fig==None:
			fig=plt.figure()
		if ax==None:
			ax=fig.add_subplot(111)
			
		if model is not None:
			try:
				if m.prof.head["model_number"]!=model:
					m.loadProfile(num=int(model))
			except:
				m.loadProfile(num=int(model))

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
		self._setYLim(ax,ax.get_ylim(),y1rng,rev=y1rev,log=y1log)

		if show_burn:
			self._plotBurnRegions(m,ax,px,py,show_line=show_burn_line,show_x=show_burn_x,ind=mInd)

		if show_mix:
			self._plotMixRegions(m,ax,px,py,show_line=show_mix_line,show_x=show_mix_x,ind=mInd)
	
		if show_burn or show_mix:
			self._showBurnMixLegend(ax,burn=show_burn,mix=show_mix)

		if show_core_loc:
			self._plotCoreLoc(m,ax,xaxis,px,ax.get_ylim()[0],ax.get_ylim()[1])
	
	
		y2_is_valid=False
		if y2 is not None:
			ax2=ax.twinx()
			try:
				y=m.prof.data[y2][mInd]
				px,py=self._plotAnnotatedLine(ax2,x[mInd],y,fy2,xrngL[0],xrngL[1],y2rng[0],y2rng[1],
										annotate_line=False,label=self.safeLabel(y2label,y2),
										points=points,xlog=xlog,ylog=y2log,xrev=xrev,
										yrev=y2rev,linecol=y2col)  
	
				if y2Textcol is None:
					y2labcol=y2col
				else:
					y2labcol=y2Textcol
				
				ax2.set_ylabel(self.safeLabel(y2label,y2), color=y2labcol)
				if show_burn_2:
					self._plotBurnRegions(m,ax2,px,py,show_line=show_burn_line,show_x=show_burn_x,ind=mInd)
				if show_mix_2:
					self._plotMixRegions(m,ax2,px,py,show_line=show_mix_line,show_x=show_mix_x,ind=mInd)
				
				y2_is_valid=True
			except:
				pass

		self._setTicks(ax)
		

		if xlabel is not None:
			ax.set_xlabel(xlabel)
		else:
			l=self.labels(xaxis)
			if l is not None:
				ax.set_xlabel(l)
			else:
				ax.set_xlabel(xaxis.replace('_',' '))
			
		if y1label is not None:
			ax.set_ylabel(y1label)
		else:
			ax.set_ylabel(y1.replace('_',' '), color=y1col)
			
		if y2 is not None and y2_is_valid:
			if y2label is not None:
				ax2.set_ylabel(y2label)
			else:
				ax2.set_ylabel(y2.replace('_',' '), color=y2col)
				
		self.setTitle(ax,show_title_name,show_title_model,show_title_age,title_name,m.prof.head["model_number"],m.prof.head["star_age"])
		
		
		if show:
			plt.show()

	def plotHistory(self,m,xaxis='model_number',y1='star_mass',y2=None,show=True,
					ax=None,xmin=None,xmax=None,xlog=False,y1log=False,
					y2log=False,y1col='b',y2col='r',minMod=0,maxMod=-1,xrev=False,
					y1rev=False,y2rev=False,points=False,xlabel=None,y1label=None,
					y2label=None,fig=None,y1rng=[None,None],y2rng=[None,None],
					fx=None,fy1=None,fy2=None,show_core=False):
		
		if fig==None:
			fig=plt.figure()
		if ax==None:
			ax=fig.add_subplot(111)
		
		if maxMod<0:
			maxMod=m.hist.data["model_number"][-1]
		modelIndex=(m.hist.data["model_number"]>=minMod)&(m.hist.data["model_number"]<=maxMod)
		
		x,xrngL,mInd=self._setXAxis(m.hist.data[xaxis][modelIndex],xmin,xmax,fx)
			
		y=m.hist.data[y1][modelIndex][mInd]
		self._plotAnnotatedLine(ax=ax,x=x[mInd],y=y,fy=fy1,xmin=xrngL[0],xmax=xrngL[1],
								ymin=y1rng[0],ymax=y1rng[1],annotate_line=False,
								label=self.safeLabel(y1label,y1),points=points,
								xlog=xlog,ylog=y1log,xrev=xrev,yrev=y1rev,linecol=y1col)
		
		self._setYLim(ax,ax.get_ylim(),y1rng,rev=y1rev,log=y1log)


		if y2 is not None:
			try:
				ax2 = ax.twinx()
				y=m.hist.data[y2][modelIndex][mInd]
				self._plotAnnotatedLine(ax,x[mInd],y,fy2,xrngL[0],xrngL[1],y2rng[0],
										y2rng[1],annotate_line=False,
										label=self.safeLabel(y1label,y1),points=points,
										xlog=xlog,ylog=y2log,xrev=xrev,yrev=y2rev,linecol=y2col)  
			except:
				pass


		if xlabel is not None:
			ax.set_xlabel(xlabel)
		else:
			l=self.labels(xaxis)
			if l is not None:
				ax.set_xlabel(l)
			else:
				ax.set_xlabel(xaxis.replace('_',' '))
			
		if y1label is not None:
			ax.set_ylabel(y1label)
		else:
			if y2 is None:
				y1textcol='k'
			else:
				y1textcol=y1col
			ax.set_ylabel(y1.replace('_',' '), color=y1textcol)
			
		if y2 is not None:
			if y2label is not None:
				ax2.set_ylabel(y2label)
			else:
				ax2.set_ylabel(y2.replace('_',' '), color=y2col)
				
		if show_core:
			self._showMassLocHist(m,fig,ax,x,y,mInd)
		
		if show:
			plt.show()

	def plotKip(self,m,show=True,reloadHistory=False,xaxis='num',ageZero=0.0,ax=None,xrng=[-1,-1],mix=None,
				cmin=None,cmax=None,burnMap=[mpl.cm.Purples_r,mpl.cm.hot_r],fig=None,yrng=None,
				show_mass_loc=False,show_mix_labels=True,mix_alpha=1.0,step=1,y2=None):
		if fig==None:
			fig=plt.figure()
			
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
		modInd[::step]=True
		
		if xrng[0]>=0:
			modInd=modInd&(m.hist.data["model_number"]>=xrng[0])&(m.hist.data["model_number"]<=xrng[1])
			
		
		#modDiff=np.diff(m.hist.data["model_number"][modInd])
		#if np.all(modDiff) !=modDiff[0]:
			#raise ValueError("model_number must be monotomically increasing, ie set history_interval=1")

			
		q=np.linspace(0.0,np.max(m.hist.data["star_mass"]),np.max(m.hist.data["num_zones"][modInd]))
		numModels=np.count_nonzero(modInd)

		burnZones=np.zeros((numModels,np.size(q)))
			
		self.numMixZones=int([x.split('_')[2] for  x in m.hist.data.dtype.names if "mix_qtop" in x][-1])
		self.numBurnZones=int([x.split('_')[2] for x in m.hist.data.dtype.names if "burn_qtop" in x][-1])
		#print self.numMixZones
		for j in range(1,self.numBurnZones+1):
			if m.hist.data["burn_qtop_"+str(j)][-1] == -1:
				self.numBurnZones=j
				break
				
		for j in range(1,self.numMixZones+1):
			if m.hist.data["mix_qtop_"+str(j)][0] == 1.0:
				self.numMixZones=j
				break

		k=0		
		for jj in m.hist.data["model_number"][modInd]:
			ind2b=np.zeros(np.size(q),dtype='bool')
			i=m.hist.data["model_number"]==jj
			for j in range(1,self.numBurnZones+1):
				indb=(q<= m.hist.data["burn_qtop_"+str(j)][i]*m.hist.data['star_mass'][i])&np.logical_not(ind2b)
				burnZones[k,indb]=m.hist.data["burn_type_"+str(j)][i]
				ind2b=ind2b|indb
				if m.hist.data["burn_qtop_"+str(j)][i] ==1.0:
					break
			k=k+1

		#age=np.log10((m.hist.data["star_age"]-ageZero)/10**6)
		Xmin=m.hist.data["model_number"][modInd][0]
		Xmax=m.hist.data["model_number"][modInd][-1]
		#Xmin=age[0]
		#Xmax=age[-1]
			
		Ymin=q[0]
		Ymax=q[-1]
		#XX,YY=np.meshgrid(np.linspace(Xmin,Xmax,numModels),q)
		#XX,YY=np.meshgrid(m.hist.data["model_number"][modInd],q)
		extent=(Xmin,Xmax,Ymin,Ymax)
		
		burnZones[burnZones<-100]=0.0

		#bb=np.zeros(np.shape(burnZones))
		#try:
			#bb[burnZones<0]=-burnZones[burnZones<0]/np.min(burnZones[(burnZones<0)])
		#except:
			#pass
		#try:
			#bb[burnZones>0]=burnZones[burnZones>0]/np.max(burnZones[(burnZones>0)])
		#except:
			#pass
		##burnZones=0.0
		
		##ax.imshow(bb.T,cmap=plt.get_cmap('seismic'),vmin=-1.0,vmax=1.0,extent=extent,interpolation='nearest',origin='lower',aspect='auto')
		##ax.contourf(XX,YY,bb.T,cmap=plt.get_cmap('seismic'),vmin=-1.0,vmax=1.0,origin='lower')
		#b1=np.copy(burnZones.T)
		#b2=np.copy(burnZones.T)
		#b1[b1<0.0]=np.nan
		#b2[b2>0.0]=np.nan
		
		#im2=ax.imshow(b2,cmap=plt.get_cmap('Purples_r'),extent=extent,interpolation='nearest',origin='lower',aspect='auto')
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

		im1=ax.imshow(burnZones.T,cmap=newCm,extent=extent,interpolation='nearest',
						origin='lower',aspect='auto',vmin=vmin,vmax=vmax)		
		burnZones=0

		
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
				if m.hist.data["mix_qtop_"+str(j)][i]==1.0:
					break
			k=k+1					
				
		mixZones[mixZones==0]=-np.nan
		
		mixCmap,mixNorm=self._setMixRegionsCol(kip=True)
		
		ax.imshow(mixZones.T,cmap=mixCmap,norm=mixNorm,extent=extent,interpolation='nearest',origin='lower',aspect='auto',alpha=mix_alpha)
		#ax.contourf(XX,YY,mixZones.T,cmap=cmap,norm=norm,origin='lower')
		mixZones=0
		plt.xlabel(r"$\rm{Model\; number}$")
		plt.ylabel(r"$\rm{Mass}\; [M_{\odot}]$")
		
		cb=plt.colorbar(im1)
		cb.solids.set_edgecolor("face")

		cb.set_label(r'$\rm{sign}\left(\epsilon_{\rm{nuc}}-\epsilon_{\nu}\right)\log_{10}\left(\rm{max}\left(1.0,|\epsilon_{\rm{nuc}}-\epsilon_{\nu}|\right)\right)$')
		fig.set_size_inches(12,9.45)
		
		#ax.locator_params(nbins=6)
		#self._setTicks(ax)
		#ax.set_tick_params(axis='both',which='both')
		self._setYLim(ax,ax.get_ylim(),yrng)
		
		#Add line at outer mass location
		ax.plot(m.hist.data['model_number'][modInd],m.hist.data['star_mass'][modInd],c='k')
		
		lt=np.log10((m.hist.data['star_age'])*3600.0*24.0*365.0)
		
		if y2 is not None:
			ax2=ax.twinx()
			ax2.plot(m.hist.data['model_number'][modInd],m.hist.data[y2][modInd],c='k')
		
		
		if show_mass_loc:
			self._showMassLoc(m,fig,ax,np.linspace(Xmin,Xmax,np.count_nonzero(modInd)),modInd)
		
		if show:
			plt.show()
		

	def plotKip2(self,m,show=True,reloadHistory=False,xaxis='num',ageZero=0.0,ax=None,xrng=[-1,-1],mix=None,
				cmin=None,cmax=None,burnMap=[mpl.cm.Purples_r,mpl.cm.hot_r],fig=None,yrng=None,
				show_mass_loc=False,show_mix_labels=True,mix_alpha=1.0,step=1,max_mass=99999.0,age_collapse=False,age_log=True,age_reverse=False,
				mod_out=None,megayears=False,xlabel=None):
		if fig==None:
			fig=plt.figure()
			
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
		
		#modInd=modInd&(m.hist.model_number>=m.hist.model_number[m.hist.burn_type_1<0.0][0])
		print(m.hist.model_number[modInd])
		
		#Age in years does not have enogh digits to be able to distingush the final models in pre-sn progenitors
		age=np.cumsum(10**np.longdouble(m.hist.log_dt))
		if age_collapse:
			age=age[-1]-age
			#Fudge the last value not to be exactly 0.0
			age[-1]=(age[-2]/2.0)
		
		age=age[modInd]
		
		if megayears:
			age=age/10**6
		
		if age_log:
			age=np.log10(age)
		
		if age_reverse:
			age=age[::-1]
			
		print(age)
		
		q=np.linspace(0.0,np.minimum(max_mass,np.max(m.hist.data["star_mass"])),np.max(m.hist.data["num_zones"][modInd]))
		numModels=np.count_nonzero(modInd)

		burnZones=np.zeros((numModels,np.size(q)))
			
		self.numMixZones=int([x.split('_')[2] for  x in m.hist.data.dtype.names if "mix_qtop" in x][-1])
		self.numBurnZones=int([x.split('_')[2] for x in m.hist.data.dtype.names if "burn_qtop" in x][-1])
		#print self.numMixZones
		for j in range(1,self.numBurnZones+1):
			if m.hist.data["burn_qtop_"+str(j)][-1] == -1:
				self.numBurnZones=j
				break
				
		for j in range(1,self.numMixZones+1):
			if m.hist.data["mix_qtop_"+str(j)][0] == 1.0:
				self.numMixZones=j
				break

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
		
		#print(age[-1],age[-2])
		ageGrid=np.linspace(age[0],age[-1],1000)
		massGrid=np.linspace(0.0,np.minimum(max_mass,np.max(m.hist.data['star_mass'])),1000)
		
		grid_xin,grid_yin=np.meshgrid(age,q,indexing='ij')
		grid_x,grid_y=np.meshgrid(ageGrid,massGrid,indexing='ij')

		grid_xin=[]
		grid_yin=[]
		grid_data=[]
		
		bt=burnZones.T
		for i in range(len(q)):
			for j in range(len(age)):
				grid_xin.append(age[j])
				grid_yin.append(q[i])
				grid_data.append(bt[i,j])
				
		burnZones=0.0
		grid_xin=np.array(grid_xin)
		grid_yin=np.array(grid_yin)
		grid_data=np.array(grid_data)
		
		grid_z=interpolate.griddata((grid_xin,grid_yin),grid_data,(grid_x,grid_y),method='nearest')
		#print(grid_z)
		grid_z=np.double(grid_z)
		extent=(ageGrid[0],ageGrid[-1],Ymin,Ymax)
		extent=np.double(np.array(extent))

		if cmin is None:
			vmin=np.nanmin(grid_data)
		else:
			vmin=cmin
			
		if cmax is None:
			vmax=np.nanmax(grid_data)
		else:
			vmax=cmax
			
		if vmin < 0:
			vmax=np.maximum(np.abs(vmax),np.abs(vmin))
			vmin=-vmax
			newCm=self.mergeCmaps(burnMap,[[0.0,0.5],[0.5,1.0]])
		else:
			vmin=0
			newCm=burnMap[-1]

		im1=ax.imshow(grid_z.T,cmap=newCm,extent=extent,interpolation='nearest',origin='lower',aspect='auto',vmin=vmin,vmax=vmax)		
		#burnZones=0

		
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
				
		mixZones[mixZones==0]=-np.nan
		
		mixCmap,mixNorm=self._setMixRegionsCol(kip=True)
		
		grid_data=[]
		mt=mixZones.T
		for i in range(len(q)):
			for j in range(len(age)):
				grid_data.append(mt[i,j])
				
		mixZones=0.0
		grid_data=np.array(grid_data)
		
		grid_z=interpolate.griddata((grid_xin,grid_yin),grid_data,(grid_x,grid_y),method='nearest')
		#print(grid_z)
		grid_z=np.double(grid_z)
		
		ax.imshow(grid_z.T,cmap=mixCmap,norm=mixNorm,extent=extent,interpolation='nearest',origin='lower',aspect='auto',alpha=mix_alpha)
		##ax.contourf(XX,YY,mixZones.T,cmap=cmap,norm=norm,origin='lower')
		#mixZones=0
		ax.set_ylabel(r"$\rm{Mass}\; [M_{\odot}]$")
		
		cb=plt.colorbar(im1)
		cb.solids.set_edgecolor("face")

		cb.set_label(r'$\rm{sign}\left(\epsilon_{\rm{nuc}}-\epsilon_{\nu}\right)\log_{10}\left(\rm{max}\left(1.0,|\epsilon_{\rm{nuc}}-\epsilon_{\nu}|\right)\right)$')
		fig.set_size_inches(12,9.45)
		
		#self._setYLim(ax,ax.get_ylim(),yrng)
		
		##Add line at outer mass location
		#ax.plot(m.hist.data['model_number'][modInd],m.hist.data['star_mass'][modInd],c='k')
		
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
		
		ageGrid=0
		massGrid=0
		
		grid_x=0
		grid_y=0.0
		
		grid_xin=[]
		grid_yin=[]
		grid_data=[]
		
		if show_mass_loc:
			self._showMassLoc(m,fig,ax,np.linspace(Xmin,Xmax,np.count_nonzero(modInd)),modInd)
		
		self._setTicks(ax)
		
		
		if show:
			plt.show()
		
		
		
		
	def plotTRho(self,m,model=None,show=True,ax=None,xmin=-4.0,xmax=10.0,fig=None,yrng=[3.0,10.0],
				show_burn=False,show_mix=False,
				showAll=False,showBurn=False,showPgas=False,showDegeneracy=False,
				showGamma=False,showEOS=False,logT=False,logRho=False,
				ycol='k'):
		if fig==None:
			fig=plt.figure()
		if ax==None:
			ax=fig.add_subplot(111)
			
			
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

	def plotMultiProfiles(self,m,mods=None,index=None,xaxis='mass',y1='',show=True,
							ax=None,xmin=None,xmax=None,xlog=False,y1log=False,
							cmap=plt.cm.gist_ncar,xrev=False,y1rev=False,
							points=False,xlabel=None,y1label=None,fig=None,
							show_mix=False,show_burn=True):
		"""Plots mulitple profiles either given as a list of mod numbers or an index over the history data"""
		if fig==None:
			fig=plt.figure()
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
		fig=plt.figure()
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
		



