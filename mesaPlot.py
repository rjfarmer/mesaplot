#Copyright (c) 2015, Robert Farmer rjfarmer@asu.edu

#Permission to use, copy, modify, and/or distribute this software for any
#purpose with or without fee is hereby granted, provided that the above
#copyright notice and this permission notice appear in all copies.

#THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
#WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
#MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
#ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
#ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
#OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.


from __future__ import print_function
import numpy as np
import mmap
import matplotlib as mpl
#mpl.use('GTK3Cairo')
import matplotlib.pyplot as plt
import bisect
import scipy.interpolate as interpolate
from matplotlib.ticker import MaxNLocator,AutoMinorLocator
import os
import matplotlib as mat
from matplotlib import rc
import matplotlib.cm as cm
import matplotlib
from matplotlib.widgets import Button
import random

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rc('font',size=32)
rc('xtick', labelsize=28) 
rc('ytick', labelsize=28) 
mat.rcParams['axes.linewidth'] = 2.0
mat.rcParams['xtick.major.size']=18      # major tick size in points
mat.rcParams['xtick.minor.size']=9      # minor tick size in points
mat.rcParams['ytick.major.size']=18      # major tick size in points
mat.rcParams['ytick.minor.size']=9      # minor tick size in points

mat.rcParams['xtick.major.width']=0.8      # major tick size in points
mat.rcParams['xtick.minor.width']=0.6      # minor tick size in points
mat.rcParams['ytick.major.width']=0.8      # major tick size in points
mat.rcParams['ytick.minor.width']=0.6      # minor tick size in points

def randf(N,a,b):
	return a + (b - a) * (np.random.random_integers(N) - 1) / (N - 1.)

class MESA():
	hist_head=""
	hist_dat=""
	prof_head=""
	prof_dat=""
	prof_ind=""
	log_fold=""
	
	def loadHistory(self,f=""):
		"""
		Reads a MESA history file.
		
		Optional:
		f: Folder in which history.data exists, if not present uses self.log_fold, if thats
		not set trys the folder LOGS/
		
		Returns:
		self.hist_head: The header data in the history file as a structured dtype
		self.hist_dat:  The data in the main body of the histor file as a structured dtype
		self.hist_head: List of names of the header fields
		self.hist_head: List of names of the data fields
		
		Note it will clean the file up of bakups,retries and restarts, prefering to use
		the newest data line.
		"""
		if len(f)==0:
			if len(self.log_fold)==0:
				self.log_fold='LOGS/'
			f=self.log_fold
		else:
			self.log_fold=f
			
		filename=f+"/history.data"
		numLines=self._filelines(filename)
		self.hist_head=np.genfromtxt(filename,skip_header=1,skip_footer=numLines-4,names=True)
		self.hist_dat=np.genfromtxt(filename,skip_header=5,names=True,invalid_raise=False)
		self.hist_head_names=self.hist_head.dtype.names
		self.hist_dat_names=self.hist_dat.dtype.names
			
	#Inspired by http://www.mesastar.org/tools-utilities/python-based-stuff/history-log-scrubber/view
	#to remove bad lines
		while (np.any(np.diff(self.hist_dat["star_age"])<0.0)):
			ind=np.zeros(np.size(self.hist_dat["model_number"]))
			rev=np.copy(self.hist_dat["star_age"][::-1])
			diff=np.diff(rev)
			ind[0:-1]=diff<0
			ind[-1]=1
			self.hist_dat=self.hist_dat[ind[::-1].astype('bool')]
		
	def scrubHistory(self,f="",fileOut="LOGS/history.data.scrubbed"):
		self.loadHistory(f)
		with open(fileOut,'w') as f:
			print(' '.join([str(i) for i in range(1,np.size(self.hist_head_names)+1)]),file=f)
			print(' '.join([str(i) for i in self.hist_head_names]),file=f)
			print(' '.join([str(self.hist_head[i]) for i in self.hist_head_names]),file=f)
			print(" ",file=f)
			print(' '.join([str(i) for i in range(1,np.size(self.hist_dat_names)+1)]),file=f)
			print(' '.join([str(i) for i in self.hist_dat_names]),file=f)
			for j in range(np.size(self.hist_dat)):
				print(' '.join([str(self.hist_dat[i][j]) for i in self.hist_dat_names]),file=f)	
	
		
	def loadProfile(self,f='',num=None,prof=-1,mode='nearest'):
		if num==None:
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
			
	def loadMod(self,filename=None):
		"""
		Fails to read a MESA .mod file.
		"""
		try:
			from StringIO import StringIO
		except ImportError:
			from io import StringIO
		count=0
		with open(filename,'r') as f:
			for l in f:
				count=count+1
				if '!' not in l:
					break
			self.mod_head=[]
			self.mod_head_names=[]
			self.mod_head.append(int(l.split()[0]))
			self.mod_head_names.append('mod_version')
			#Blank line
			f.readline()
			count=count+1
			#Gap between header and main data
			for l in f:
				count=count+1
				if l=='\n':
					break
				self.mod_head.append(l.split()[1])
				self.mod_head_names.append(l.split()[0])
			self.mod_dat_names=[]
			l=f.readline()
			count=count+1
			self.mod_dat_names.append('zone')
			self.mod_dat_names.extend(l.split())
			self.mod_dat=np.genfromtxt(filename,skip_header=count,names=self.mod_dat_names,skip_footer=8,dtype=None)
			#Genfromtxt fails in converting 1d1 to 1.0
			for i in range(np.size(self.mod_dat)):
				for j in range(1,len(self.mod_dat_names)):
					self.mod_dat[i][j]=str(self.mod_dat[i][j]).replace('D','E')

			newDtype=np.dtype([(name,'float') for name in self.mod_dat_names])
			self.mod_dat=self.mod_dat.astype(newDtype)

		
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
			elif len(rng)>=2 and rng[0]>0:
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
		self.prof_head: The header data in the profile as a structured dtype
		self.prof_dat:  The data in the main body of the profile file as a structured dtype
		self.prof_head: List of names of the header fields
		self.prof_head: List of names of the data fields
		"""
		numLines=self._filelines(filename)
		try:
			self.prof_head=np.genfromtxt(filename,skip_header=1,skip_footer=numLines-4,names=True)
			self.prof_dat=np.genfromtxt(filename,skip_header=5,names=True,invalid_raise=False)
			self.prof_head_names=self.prof_head.dtype.names
			self.prof_dat_names=self.prof_dat.dtype.names
		except:
			pass
				
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
		
	def _getNextProfile(self):
		currMod=self.prof_head["model_number"]
		itemindex = np.where(self.prof_ind["model"]==currMod)[0]
		try:
			nextItem=self.prof_ind["model"][itemindex[0]+1]
		except IndexError:
			return None
		itemindex = np.where(self.prof_ind["model"]==nextItem)
		if np.shape(itemindex[0])[0]==0:
			return None
		else:
			return nextItem
	
	def _getPrevProfile(self):
		currMod=self.prof_head["model_number"]
		itemindex = np.where(self.prof_ind["model"]==currMod)[0]
		try:
			nextItem=self.prof_ind["model"][itemindex[0]-1]
		except IndexError:
			return None
		itemindex = np.where(self.prof_ind["model"]==nextItem)
		if np.shape(itemindex[0])[0]==0:
			return None
		else:
			return nextItem
	
	def _getNextModNumHist(self):
		pass

	def _getPrevModNumHist(self):
		pass

class plot():
	def labels(self,label,log=False,center=False):
		l=''
		if log or 'log' in label:
			l=r'$\log_{10}\;$'
	
		if label=='mass':
			l=l+r"$\rm{Mass}\; [M_{\odot}]$"
		if label=='model':
			l=l+r"$\rm{Model\; number}$"
		if 'teff' in label or label=='logT':
			if center:
				l=l+r"$T_{eff,c}\; [K]$"
			else:
				l=l+r"$T_{eff}\; [K]$"
		if label=='rho':
			if center:
				l=l+r"$\rho_{c}\; [K]$"
			else:
				l=l+r"$\rho\; [K]$"
		if label=='log_column_depth':
			l=l+r'$y\; [\rm{g}\; \rm{cm}^{-2}]$'
		elif 'lum' in label:
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
		
	def _listAbun(self,m):
		abun_list=[]
		for i in m.prof_dat.dtype.names:
			if len(i)<=5 and len(i)>=2:
				if i[0].isalpha() and (i[1].isalpha() or i[1].isdigit()) and any(char.isdigit() for char in i) and i[-1].isdigit():
					if (len(i)==5 and i[-1].isdigit() and i[-2].isdigit()) or len(i)<5:
						abun_list.append(i)
		return abun_list
		
	def _listAbunHistory(self,m):
		abun_list=[]
		for j in m.hist_dat.dtype.names:
			i=j.split('_')[-1]
			if len(i)<=5 and len(i)>=2:
				if i[0].isalpha() and (i[1].isalpha() or i[1].isdigit()) and any(char.isdigit() for char in i) and i[-1].isdigit():
					if (len(i)==5 and i[-1].isdigit() and i[-2].isdigit()) or len(i)<5:
						abun_list.append(i)
		return abun_list
		
	def _listBurn(self,m):
		burnList=[]
		extraBurn=["pp","cno","tri_alfa","c12_c12","c12_O16","o16_o16","pnhe4","photo","other"]
		for i in m.prof_dat.dtype.names:
			if "burn_" in i or i in extraBurn:
				burnList.append(i)
		return burnList
		
	def _listBurnHistory(self,m):
		burnList=[]
		extraBurn=["pp","cno","tri_alfa","c12_c12","c12_O16","o16_o16","pnhe4","photo","other"]
		for i in m.hist_dat.dtype.names:
			if "burn_" in i or i in extraBurn:
				burnList.append(i)
		return burnList


	def _setMixRegionsCol(self):
		cmap = mpl.colors.ListedColormap([[0.18, 0.545, 0.34], [0.53, 0.808, 0.98],
			[0.96, 0.96, 0.864], [0.44, 0.5, 0.565],[0.8, 0.6, 1.0],
			[0.0, 0.4, 1.0],[1.0, 0.498, 0.312],[0.824, 0.705, 0.55]])

		cmap.set_over((1., 1., 1.))
		cmap.set_under((0., 0., 0.))
		#bounds = [-0.01,0.99,1.99,2.99,3.99,4.99,5.99,6.99,7.99,8.99]
		bounds=[-1,0,1,2,3,4,5,6,7,8,9]
		norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
		return cmap,norm
		
	def _setTicks(self,ax):
		ax.yaxis.set_major_locator(MaxNLocator(5))
		ax.xaxis.set_major_locator(MaxNLocator(5))
		ax.yaxis.set_minor_locator(AutoMinorLocator(10))
		ax.xaxis.set_minor_locator(AutoMinorLocator(10))
	
	def _plotBurnRegions(self,m,ax,x,y,show_x,show_line,yrng=None):
		# non 0.0, yellow 1, ornage 10**4, red 10**7
		ylim=ax.get_ylim()
		
		if show_x:
			yy=np.zeros(np.size(x))
			if yrng is not None:
				yy[:]=yrng[0]
			else:
				yy[:]=ylim[0]
			size=240
		if show_line:
			yy=y
			size=180
		
		ind=(m.prof_dat['net_nuclear_energy']>=1.0)&(m.prof_dat['net_nuclear_energy']<=4.0)	
		ax.scatter(x[ind],yy[ind],c='yellow',s=size,linewidths=0,alpha=1.0)
		ind=(m.prof_dat['net_nuclear_energy']>=4.0)&(m.prof_dat['net_nuclear_energy']<=7.0)
		ax.scatter(x[ind],yy[ind],c='orange',s=size,linewidths=0,alpha=1.0)
		ind=(m.prof_dat['net_nuclear_energy']>=7.0)
		ax.scatter(x[ind],yy[ind],c='red',s=size,edgecolor='none',alpha=1.0)
		
		ax.set_ylim(ylim)
		
		
	def _plotMixRegions(self,m,ax,x,y,show_x,show_line,yrng=None):
		ylim=ax.get_ylim()
		
		if show_x:
			yy=np.zeros(np.size(x))
			if yrng is not None:
				yy[:]=yrng[0]
			else:
				yy[:]=ylim[0]
			size=90
		if show_line:
			yy=y
			size=60
	
		cmap,norm=self._setMixRegionsCol()
		
		isSet=None
		for mixLabel in ['mixing_type','conv_mixing_type']:
			try:
				col=m.prof_dat[mixLabel]
				isSet=True
				break
			except:
				continue
			
		if isSet is None:
			raise(ValueError,"Need mixing type in profile file for showing mix regions, either its mixing_type or conv_mixing_type")
		
		ax.scatter(x,yy,c=col,s=size,cmap=cmap,norm=norm,linewidths=0)
	
		ax.set_ylim(ylim)
	
	def _annotateLine(self,m,ax,x,y,num_labels,xmin,xmax,text,line,fontsize=mat.rcParams['font.size']-12):
		for ii in range(1,num_labels+1):
			ind=(x>=xmin)&(x<=xmax)
			f = interpolate.interp1d(x[ind],y[ind])
			xp1=((xmax-xmin)*(ii/(num_labels+1.0)))+xmin
			yp1=f(xp1)
			ax.annotate(text, xy=(xp1,yp1), xytext=(xp1,yp1),color=line.get_color(),fontsize=fontsize)
	
	def plotAbun(self,m,model=None,show=True,ax=None,xaxis='mass',xmin=None,xmax=None,yrng=[-3.0,1.0],
						cmap=plt.cm.gist_ncar,num_labels=3,xlabel=None,points=False,abun=None,abun_random=False,
					show_burn=False,show_mix=False):
		if ax ==None:
			fig=plt.figure()
			ax=fig.add_subplot(111)
		#m.loadProfile(num=int(model))
		
		if model is not None:
			m.loadProfile(num=int(model))
			
		xrngL=[0,0]
		if xmin is not None:
			xrngL[0]=xmin
		else:
			xrngL[0]=np.min(m.prof_dat[xaxis])

		if xmax is not None:
			xrngL[1]=xmax
		else:
			xrngL[1]=np.max(m.prof_dat[xaxis])

		if abun is None:
			abun_list=self._listAbun(m)
		else:
			abun_list=abun
			
		num_plots=len(abun_list)
		#Helps when we have many elements not on the plot that stretch the colormap
		if abun_random:
			random.shuffle(abun_list)

		plt.gca().set_color_cycle([cmap(i) for i in np.linspace(0.0,0.9,num_plots)])
		
		for i in abun_list:
			y=np.log10(m.prof_dat[i])
			y[np.logical_not(np.isfinite(y))]=yrng[0]-(yrng[1]-yrng[0])
			line, =ax.plot(m.prof_dat[xaxis],y,label=i,linewidth=2)
			if points:
				ax.scatter(m.prof_dat[xaxis],y)
				
			self._annotateLine(m,ax,m.prof_dat[xaxis],y,num_labels,xrngL[0],xrngL[1],i,line)

		if show_burn:
			self._plotBurnRegions(m,ax,m.prof_dat[xaxis],y,show_line=False,show_x=True,yrng=yrng)

		if show_mix:
			self._plotMixRegions(m,ax,m.prof_dat[xaxis],y,show_line=False,show_x=True,yrng=yrng)
			
		self._setTicks(ax)
		#ax.legend(loc=0,fontsize=20)
		ax.set_xlabel(self.safeLabel(xlabel,xaxis))

		ax.set_ylabel(r'$\log_{10}$ Abundance')
		ax.set_ylim(yrng)
		ax.set_xlim(xrngL)
		ax.set_title("Abundance")
		if show:
			plt.show()
			

	def plotDynamo(self,m,xaxis='mass',model=None,show=True,ax=None,xmin=None,xmax=None,xlabel=None,yrng=[0.0,10.0],
						show_burn=False,show_mix=False):
		if ax ==None:
			fig=plt.figure()
			ax=fig.add_subplot(111)
	
		if model is not None:
			m.loadProfile(num=int(model))
			
		xrngL=[0,0]
		if xmin is not None:
			xrngL[0]=xmin
		else:
			xrngL[0]=np.min(m.prof_dat[xaxis])

		if xmax is not None:
			xrngL[1]=xmax
		else:
			xrngL[1]=np.max(m.prof_dat[xaxis])
		
		#ind=(m.prof_dat['dynamo_log_B_r']>-90)
		ax.plot(m.prof_dat[xaxis][ind],m.prof_dat['dynamo_log_B_r'],label=r'$B_r$',linewidth=2)
		#ind=mInd&(m.prof_dat['dynamo_log_B_phi']>-90)
		ax.plot(m.prof_dat[xaxis][ind],m.prof_dat['dynamo_log_B_phi'],label=r'$B_{\phi}$',linewidth=2)

		if show_burn:
			self._plotBurnRegions(m,ax,m.prof_dat[xaxis],y,show_line=False,show_x=True)

		if show_mix:
			self._plotMixRegions(m,ax,m.prof_dat[xaxis],y,show_line=False,show_x=True)
		
		try:
			ax.legend(loc=0)
		except:
			pass
		ax.set_xlabel(self.safeLabel(xlabel,xaxis))
		self._setTicks(ax)
		ax.set_xlim(xrngL)
		ax.set_ylim(yrng)
		ax.set_title("Dynamo Model")
		if show:
			plt.show()

	def plotAngMom(self,m,xaxis='mass',model=None,show=True,ax=None,xmin=None,xmax=None,xlabel=None,yrng=[0.0,10.0],
						show_burn=False,show_mix=False):
		if ax ==None:
			fig=plt.figure()
			ax=fig.add_subplot(111)
			
		if model is not None:
			m.loadProfile(num=int(model))
		xrngL=[0,0]
		if xmin is not None:
			xrngL[0]=xmin
		else:
			xrngL[0]=np.min(m.prof_dat[xaxis])

		if xmax is not None:
			xrngL[1]=xmax
		else:
			xrngL[1]=np.max(m.prof_dat[xaxis])

		for i in m.prof_dat.dtype.names:
			if "am_log_D" in i:
				#ind=mInd&(m.prof_dat[i]>-90.0)
				ax.plot(m.prof_dat[xaxis][ind],m.prof_dat[i][ind],label=r"$D_{"+i.split('_')[3]+"}$")

		if show_burn:
			self._plotBurnRegions(m,ax,m.prof_dat[xaxis],y,show_line=False,show_x=True)

		if show_mix:
			self._plotMixRegions(m,ax,m.prof_dat[xaxis],y,show_line=False,show_x=True)

		try:
			ax.legend(loc=0)
		except:
			pass

		ax.set_xlabel(self.safeLabel(xlabel,xaxis))
		ax.set_title("Ang Mom Model")
		self._setTicks(ax)
		ax.set_xlim(xrngL)
		ax.set_ylim(yrng)
		
		if show:
			plt.show()
			
	def plotBurn(self,m,xaxis='mass',model=None,show=True,ax=None,xmin=None,xmax=None,xlabel=None,
					cmap=plt.cm.gist_ncar,yrng=[0.0,10.0],num_labels=7,burn_random=False,points=False,
					show_burn=False,show_mix=False):
		if ax ==None:
			fig=plt.figure()
			ax=fig.add_subplot(111)
			
		if model is not None:
			m.loadProfile(num=int(model))
		xrngL=[0,0]
		if xmin is not None:
			xrngL[0]=xmin
		else:
			xrngL[0]=np.min(m.prof_dat[xaxis])

		if xmax is not None:
			xrngL[1]=xmax
		else:
			xrngL[1]=np.max(m.prof_dat[xaxis])



		burn_list=self._listBurn(m)
		num_plots=len(burn_list)
		
		if burn_random:
			random.shuffle(burn_list)
				
		plt.gca().set_color_cycle([cmap(i) for i in np.linspace(0.0,0.9,num_plots)])
			
		for i in burn_list:
			y=np.log10(m.prof_dat[i])
			y[np.logical_not(np.isfinite(y))]=yrng[0]-(yrng[1]-yrng[0])
			line, =ax.plot(m.prof_dat[xaxis],y,label=i.replace('_',' '))
			if points:
				ax.scatter(m.prof_dat[xaxis],y)
			self._annotateLine(m,ax,m.prof_dat[xaxis],y,num_labels,xrngL[0],xrngL[1],self.safeLabel(None,i),line)

		
		if show_burn:
			self._plotBurnRegions(m,ax,m.prof_dat[xaxis],y,show_line=False,show_x=True)

		if show_mix:
			self._plotMixRegions(m,ax,m.prof_dat[xaxis],y,show_line=False,show_x=True)
		
		self._setTicks(ax)
		ax.set_ylim(yrng)
		ax.set_xlim(xrngL)
		ax.set_xlabel(self.safeLabel(xlabel,xaxis))
		ax.set_title("Burning")
		if show:
			plt.show()

	def plotBurnSummary(self,m,xaxis='model_number',minMod=0,maxMod=-1,show=True,ax=None,xmin=None,xmax=None,xlabel=None,
					cmap=plt.cm.nipy_spectral,yrng=[0.0,10.0],num_labels=7,burn_random=False,points=False,
					show_burn=False,show_mix=False):
						
		if ax ==None:
			fig=plt.figure()
			ax=fig.add_subplot(111)
			
		if maxMod<0:
			maxMod=m.hist_dat["model_number"][-1]
		modelIndex=(m.hist_dat["model_number"]>=minMod)&(m.hist_dat["model_number"]<=maxMod)
		
		mInd=np.zeros(np.size(m.hist_dat[xaxis][modelIndex]),dtype='bool')
		mInd[:]=True
		xrngL=[m.hist_dat[xaxis][modelIndex].min(),m.hist_dat[xaxis][modelIndex].max()]
		if xmin is not None:
			mInd=(m.hist_dat[xaxis][modelIndex]>=xmin)
			xrngL[0]=xmin

		if xmax is not None:
			mInd=mInd&(m.hist_dat[xaxis][modelIndex]<=xmax)
			xrngL[1]=xmax

			
		burn_list=self._listBurnHistory(m)
		num_plots=len(burn_list)
		
		if burn_random:
			random.shuffle(burn_list)
				
		plt.gca().set_color_cycle([cmap(i) for i in np.linspace(0.0,0.9,num_plots)])
			
		for i in burn_list:
			y=m.hist_dat[i][mInd]
			
			y[np.logical_not(np.isfinite(y))]=yrng[0]-(yrng[1]-yrng[0])
			line, =ax.plot(m.hist_dat[xaxis][mInd],y)
			if points:
				ax.scatter(m.hist_dat[xaxis][mInd],y)
			self._annotateLine(m,ax,m.hist_dat[xaxis][mInd],y,num_labels,xrngL[0],xrngL[1],self.safeLabel(None,i),line)

		if show_burn:
			self._plotBurnRegions(m,ax,m.hist_dat[xaxis][mInd],y,show_line=False,show_x=True)

		if show_mix:
			self._plotMixRegions(m,ax,m.hist_dat[xaxis][mInd],y,show_line=False,show_x=True)
		
		self._setTicks(ax)
		if yrng is not None:
			ax.set_ylim(yrng)
		ax.set_xlim(xrngL)
		ax.set_xlabel(self.safeLabel(xlabel,xaxis))
		ax.set_ylabel(self.labels('log_lum'))

		if show:
			plt.show()

	def plotAbunSummary(self,m,xaxis='model_number',minMod=0,maxMod=-1,show=True,ax=None,xmin=None,xmax=None,xlabel=None,
					cmap=plt.cm.nipy_spectral,yrng=[0.0,10.0],num_labels=7,abun_random=False,points=False,
					show_burn=False,show_mix=False,abun=None):
						
		if ax ==None:
			fig=plt.figure()
			ax=fig.add_subplot(111)
			
		if maxMod<0:
			maxMod=m.hist_dat["model_number"][-1]
		modelIndex=(m.hist_dat["model_number"]>=minMod)&(m.hist_dat["model_number"]<=maxMod)
		
		mInd=np.zeros(np.size(m.hist_dat[xaxis][modelIndex]),dtype='bool')
		mInd[:]=True
		xrngL=[m.hist_dat[xaxis][modelIndex].min(),m.hist_dat[xaxis][modelIndex].max()]
		if xmin is not None:
			mInd=(m.hist_dat[xaxis][modelIndex]>=xmin)
			xrngL[0]=xmin

		if xmax is not None:
			mInd=mInd&(m.hist_dat[xaxis][modelIndex]<=xmax)
			xrngL[1]=xmax

			
		if abun is None:
			abun_list=self._listAbunHistory(m)
		else:
			abun_list=abun
			
		num_plots=len(abun_list)
		#Helps when we have many elements not on the plot that stretch the colormap
		if abun_random:
			random.shuffle(abun_list)
				
		plt.gca().set_color_cycle([cmap(i) for i in np.linspace(0.0,0.9,num_plots)])
			
		for i in abun_list:
			y=m.hist_dat["log_total_mass_"+i][mInd]
			
			y[np.logical_not(np.isfinite(y))]=yrng[0]-(yrng[1]-yrng[0])
			line, =ax.plot(m.hist_dat[xaxis][mInd],y)
			if points:
				ax.scatter(m.hist_dat[xaxis][mInd],y)
			self._annotateLine(m,ax,m.hist_dat[xaxis][mInd],y,num_labels,xrngL[0],xrngL[1],self.safeLabel(None,i),line)

		if show_burn:
			self._plotBurnRegions(m,ax,m.hist_dat[xaxis][mInd],y,show_line=False,show_x=True)

		if show_mix:
			self._plotMixRegions(m,ax,m.hist_dat[xaxis][mInd],y,show_line=False,show_x=True)
		
		self._setTicks(ax)
		if yrng is not None:
			ax.set_ylim(yrng)
		ax.set_xlim(xrngL)
		ax.set_xlabel(self.safeLabel(xlabel,xaxis))
		ax.set_ylabel(self.labels('log_abundance'))

		if show:
			plt.show()


	def plotProfile(self,m,model=None,xaxis='mass',y1='logT',y2=None,show=True,ax=None,xmin=None,xmax=None,xL='linear',y1L='linear',y2L='linear',y1col='b',
							y2col='r',xrev=False,y1rev=False,y2rev=False,points=False,xlabel=None,y1label=None,y2label=None,
							show_burn=False,show_burn_2=False,show_burn_x=False,show_burn_line=False,
							show_mix=False,show_mix_2=False,show_mix_x=False,show_mix_line=False,y1Textcol=None,y2Textcol=None):
		if ax ==None:
			fig=plt.figure()
			ax=fig.add_subplot(111)
			
		if model is not None:
			m.loadProfile(num=int(model))
		mInd=np.zeros(np.size(m.prof_dat[xaxis]),dtype='bool')
		mInd[:]=True
		xrngL=[m.prof_dat[xaxis].min(),m.prof_dat[xaxis].max()]
		if xmin is not None:
			mInd=(m.prof_dat[xaxis]>=xmin)
			xrngL[0]=xmin

		if xmax is not None:
			mInd=mInd&(m.prof_dat[xaxis]<=xmax)
			xrngL[1]=xmax

		if xL=='log':
			xrngL=np.log10(xrngL)
			
		if xrev:
			ax.set_xlim(xrngL[1],xrngL[0])
		else:
			ax.set_xlim(xrngL)


		if y1L=='log':
			y=np.log10(m.prof_dat[y1][mInd])
		else:
			y=m.prof_dat[y1][mInd]
		if xL=='log':
			x=np.log10(m.prof_dat[xaxis][mInd])
		else:
			x=m.prof_dat[xaxis][mInd]
		ax.plot(x,y,c=y1col,linewidth=2)
		if points:
			ax.scatter(x,y,c=y1col)
		
		if y1Textcol is None:
			y1labcol=y1col
		else:
			y1labcol=y1Textcol


		ax.set_ylabel(self.safeLabel(y1label,y1), color=y1labcol)
		ylim=ax.get_ylim()
		if y1rev:
			ax.set_ylim(ylim[1],ylim[0])

		if show_burn:
			self._plotBurnRegions(m,ax,x,y,show_line=show_burn_line,show_x=show_burn_x)

		if show_mix:
			self._plotMixRegions(m,ax,x,y,show_line=show_mix_line,show_x=show_mix_x)
	
		if y2 is not None:
			try:
				ax2 = ax.twinx()
				if y2L=='log':
					y=np.log10(m.prof_dat[y2][mInd])
				else:
					y=m.prof_dat[y2][mInd]
				if xL=='log':
					x=np.log10(m.prof_dat[xaxis][mInd])
				else:
					x=m.prof_dat[xaxis][mInd]
				ax2.plot(x,y,c=y2col,linewidth=2)
				if points:
					ax2.scatter(x,y,c=y2col)
					
				if y2Textcol is None:
					y2labcol=y2col
				else:
					y2labcol=y2Textcol
				
				ax2.set_ylabel(self.safeLabel(y2label,y2), color=y2labcol)
				self._setTicks(ax2)
				ylim=ax2.get_ylim()
				if y2rev:
					ax2.set_ylim(ylim[1],ylim[0])
				if show_burn_2:
					self._plotBurnRegions(m,ax2,x,y,show_line=show_burn_line,show_x=show_burn_x)
				if show_mix_2:
					self._plotMixRegions(m,ax2,x,y,show_line=show_mix_line,show_x=show_mix_x)
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
			
		if y2 is not None:
			if y2label is not None:
				ax2.set_ylabel(y2label)
			else:
				ax2.set_ylabel(y2.replace('_',' '), color=y2col)
		if show:
			plt.show()

	def plotHistory(self,m,xaxis='model_number',y1='star_mass',y2=None,show=True,ax=None,xmin=None,xmax=None,xL='linear',y1L='linear',y2L='linear',y1col='b',y2col='r',
							minMod=0,maxMod=-1,xrev=False,y1rev=False,y2rev=False,points=False,xlabel=None,y1label=None,y2label=None):
		if ax ==None:
			fig=plt.figure()
			ax=fig.add_subplot(111)
		
		if maxMod<0:
			maxMod=m.hist_dat["model_number"][-1]
		modelIndex=(m.hist_dat["model_number"]>=minMod)&(m.hist_dat["model_number"]<=maxMod)
		
		mInd=np.zeros(np.size(m.hist_dat[xaxis][modelIndex]),dtype='bool')
		mInd[:]=True
		xrngL=[m.hist_dat[xaxis][modelIndex].min(),m.hist_dat[xaxis][modelIndex].max()]
		if xmin is not None:
			mInd=(m.hist_dat[xaxis][modelIndex]>=xmin)
			xrngL[0]=xmin

		if xmax is not None:
			mInd=mInd&(m.hist_dat[xaxis][modelIndex]<=xmax)
			xrngL[1]=xmax

		if xL=='log':
			xrngL=np.log10(xrngL)
			
		if xrev:
			ax.set_xlim(xrngL[1],xrngL[0])
		else:
			ax.set_xlim(xrngL)
			
		if y1L=='log':
			y=np.log10(m.hist_dat[y1][modelIndex][mInd])
		else:
			y=m.hist_dat[y1][modelIndex][mInd]
		if xL=='log':
			x=np.log10(m.hist_dat[xaxis][modelIndex][mInd])
		else:
			x=m.hist_dat[xaxis][modelIndex][mInd]
		ax.plot(x,y,c=y1col,linewidth=2)
		if points:
			ax.scatter(x,y,c=y1col)
		ylim=ax.get_ylim()
		if y1rev:
			ax.set_ylim(ylim[1],ylim[0])


		if y2 is not None:
			try:
				ax2 = ax.twinx()
				if y2L=='log':
					y=np.log10(m.hist_dat[y2][modelIndex][mInd])
				else:
					y=m.hist_dat[y2][modelIndex][mInd]
				if xL=='log':
					x=np.log10(m.hist_dat[xaxis][modelIndex][mInd])
				else:
					x=m.hist_dat[xaxis][modelIndex][mInd]
				ax2.plot(x,y,c=y2col,linewidth=2)
				if points:
					ax2.scatter(x,y,c=y2col)
				self._setTicks(ax2)
				ylim=ax2.get_ylim()
				if y2rev:
					ax2.set_ylim(ylim[1],ylim[0])
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
		
		if show:
			plt.show()

	def plotKip(self,m,show=True,reloadHistory=False,xaxis='num',ageZero=0.0,ax=None,xrng=[-1,-1],mix=None,cmin=None,cmax=None,burnMap=[cm.Purples_r,cm.hot_r]):
		if ax ==None:
			fig=plt.figure()
			ax=fig.add_subplot(111)

		if reloadHistory:
			m.loadHistory()
			
		
		if xrng[0]>=0:
			modInd=(m.hist_dat["model_number"]>=xrng[0])&(m.hist_dat["model_number"]<=xrng[1])
		else:	
			modInd=np.zeros(np.size(m.hist_dat["model_number"]),dtype='bool')
			modInd[:]=True
			
		if np.all(np.diff(m.hist_dat["model_number"][modInd])) !=1:
			raise(ValueError,"model_number must be monotomically increasing, ie set history_interval=1")
			
		try:
			q=np.linspace(0.0,m.hist_head["initial_mass"],1*np.max(m.hist_dat["num_zones"][modInd]))
		except:
			m.loadHistory()
			q=np.linspace(0.0,m.hist_head["initial_mass"],1*np.max(m.hist_dat["num_zones"][modInd]))
		
		numModels=np.count_nonzero(modInd)

		burnZones=np.zeros((numModels,np.size(q)))
			
		self.numMixZones=int([x.split('_')[2] for  x in m.hist_dat.dtype.names if "mix_qtop" in x][-1])
		self.numBurnZones=int([x.split('_')[2] for x in m.hist_dat.dtype.names if "burn_qtop" in x][-1])
		#print self.numMixZones
		for j in range(1,self.numBurnZones+1):
			if m.hist_dat["burn_qtop_"+str(j)][-1] == -1:
				self.numBurnZones=j
				break
				
		for j in range(1,self.numMixZones+1):
			if m.hist_dat["mix_qtop_"+str(j)][0] == 1.0:
				self.numMixZones=j
				break

		k=0		
		for i in range(np.size(m.hist_dat["model_number"])):
			if modInd[i]:
				ind2b=np.zeros(np.size(q),dtype='bool')
				for j in range(1,self.numBurnZones+1):
					indb=(q<= m.hist_dat["burn_qtop_"+str(j)][i]*m.hist_dat['star_mass'][i])&np.logical_not(ind2b)
					burnZones[k,indb]=m.hist_dat["burn_type_"+str(j)][i]
					ind2b=ind2b|indb
				k=k+1

		#age=np.log10((m.hist_dat["star_age"]-ageZero)/10**6)
		Xmin=m.hist_dat["model_number"][modInd][0]
		Xmax=m.hist_dat["model_number"][modInd][-1]
		#Xmin=age[0]
		#Xmax=age[-1]
			
		Ymin=q[0]
		Ymax=q[-1]
		#XX,YY=np.meshgrid(np.linspace(Xmin,Xmax,numModels),q)
		#XX,YY=np.meshgrid(m.hist_dat["model_number"][modInd],q)
		extent=(Xmin,Xmax,Ymin,Ymax)
		
		burnZones[burnZones<-100]=0.0

		bb=np.zeros(np.shape(burnZones))
		try:
			bb[burnZones<0]=-burnZones[burnZones<0]/np.min(burnZones[(burnZones<0)])
		except:
			pass
		try:
			bb[burnZones>0]=burnZones[burnZones>0]/np.max(burnZones[(burnZones>0)])
		except:
			pass
		#burnZones=0.0
		
		#ax.imshow(bb.T,cmap=plt.get_cmap('seismic'),vmin=-1.0,vmax=1.0,extent=extent,interpolation='nearest',origin='lower',aspect='auto')
		#ax.contourf(XX,YY,bb.T,cmap=plt.get_cmap('seismic'),vmin=-1.0,vmax=1.0,origin='lower')
		b1=np.copy(burnZones.T)
		b2=np.copy(burnZones.T)
		b1[b1<0.0]=np.nan
		b2[b2>0.0]=np.nan

		
		newCm=self.mergeCmaps(burnMap,[[0.0,0.5],[0.5,1.0]])
		
		#im2=ax.imshow(b2,cmap=plt.get_cmap('Purples_r'),extent=extent,interpolation='nearest',origin='lower',aspect='auto')
		if cmin is None:
			vmin=np.nanmin(burnZones)
		else:
			vmin=cmin
			
		if cmax is None:
			vmax=np.nanmax(burnZones)
		else:
			vmax=cmax
			
		vmax=np.maximum(np.abs(vmax),np.abs(vmin))
		vmin=-vmax

		im1=ax.imshow(burnZones.T,cmap=newCm,extent=extent,interpolation='nearest',origin='lower',aspect='auto',vmin=vmin,vmax=vmax)		
		bb=0

		
		mixZones=np.zeros((numModels,np.size(q)))
		k=0
		for i in range(np.size(m.hist_dat["model_number"])):
			if modInd[i]:
				ind2=np.zeros(np.size(q),dtype='bool')
				for j in range(1,self.numMixZones+1):
					ind=(q<= m.hist_dat["mix_qtop_"+str(j)][i]*m.hist_dat['star_mass'][i])&np.logical_not(ind2)

					if mix is None:
						mixZones[k,ind]=m.hist_dat["mix_type_"+str(j)][i]
					elif mix ==-1 :
						mixZones[k,ind]=0.0
					elif m.hist_dat["mix_type_"+str(j)][i] in mix:
						mixZones[k,ind]=m.hist_dat["mix_type_"+str(j)][i]
					else:
						mixZones[k,ind]=0.0
					ind2=ind2|ind
				k=k+1					
					
		mixZones[mixZones==0]=-np.nan
		
		mixCmap,mixNorm=self._setMixRegionsCol()
		
		ax.imshow(mixZones.T,cmap=mixCmap,norm=mixNorm,extent=extent,interpolation='nearest',origin='lower',aspect='auto')
		#ax.contourf(XX,YY,mixZones.T,cmap=cmap,norm=norm,origin='lower')
		mixZones=0
		plt.xlabel(r"$\rm{Model\; number}$")
		plt.ylabel(r"$\rm{Mass}\; [M_{\odot}]$")
		
		cb=plt.colorbar(im1)
		cb.solids.set_edgecolor("face")

		cb.set_label(r'$\rm{sign}\left(\epsilon_{\rm{nuc}}-\epsilon_{\nu}\right)\log_{10}\left(\rm{max}\left(1.0,|\epsilon_{\rm{nuc}}-\epsilon_{\nu}|\right)\right)$')
		fig.set_size_inches(12,9.45)
		
		#ax.locator_params(nbins=6)
		self._setTicks(ax)
		#ax.set_tick_params(axis='both',which='both')
		
		if show:
			plt.show()
		
	def plotTRho(self,m,model=None,show=True,ax=None,xmin=None,xmax=None):
		self.plotProfile(m,xaxis='logT',y1='logRho',y1L='linear',model=model,show=show,
								xmin=xmin,xmax=xmax,ax=ax,y1col='k',xlabel=self.labels('teff',log=True),y1label=self.labels('rho',log=True))

	def plotHR(self,m,minMod=0,maxMod=-1,show=True,ax=None,xmin=None,xmax=None):
		self.plotHistory(m,xaxis='log_Teff',y1='log_L',y1L='linear',minMod=minMod,
								maxMod=maxMod,show=show,xmin=xmin,xmax=xmax,xrev=True,y1rev=True,ax=ax,y1col='k',
								xlabel=self.labels('teff',log=True),y1label=self.labels('lum',log=True))
	
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

		return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)
		
	def stackedPlots(self,m,typ='profile',num=1,model=None,xaxis='mass',show=True,fig=None,ax=None,xmin=None,xmax=None,xL='linear',xlabel=None,
								xrev=False,y1rev=[],y2rev=[],points=False,minMod=0,maxMod=-1,
								y1=[],y2=[],y1L=[],y2L=[],y1col=[],
								y2col=[],y1label=[],y2label=[]):
		if num<2:
			raise(ValueError,'num must be >=2')
		
		empty=[None]*len(y1)
		if len(y1)>0:
			if not y2:
				y2=empty
			if not y1L:
				y1L=empty
			if not y2L:
				y2L=empty
			if not y1rev:
				y1rev=empty
			if not y2rev:
				y2rev=empty
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
									y1=y1[i],y2=y2[i],y1L=y1L[i],y2L=y2L[i],y1col=y1col[i],
									y2col=y2col[i],y1label=y1label[i],y2label=y2label[i])
			else:
				self.plotHistory(m=m,xaxis=xaxis,show=False,ax=axis[i],xmin=xmin,xmax=xmax,xL=xL,xlabel=xlabel,
									xrev=xrev,y1rev=y1rev[i],y2rev=y2rev[i],points=points,
									y1=y1[i],y2=y2[i],y1L=y1L[i],y2L=y2L[i],y1col=y1col[i],
									y2col=y2col[i],y1label=y1label[i],y2label=y2label[i],minMod=minMod,maxMod=maxMod)
		
		if show:
			plt.show()

	def plotMultiProfiles(self,m,mods=None,index=None,xaxis='mass',y1='',y2='',show=True,ax=None,xmin=None,xmax=None,xL='linear',y1L='linear',cmap=plt.cm.gist_ncar,
							y2col='r',xrev=False,y1rev=False,y2rev=False,points=False,xlabel=None,y1label=None,y2label=None):
		"""Plots mulitple profiles either given as a list of mod numbers or an index over the history data"""

		if ax ==None:
			fig=plt.figure()
			ax=fig.add_subplot(111)
		
		if mods is not None:
			cm=[cmap(i) for i in np.linspace(0.0,0.9,len(mods))]
			for i in range(len(mods)):
				model=mods[i]
				self.plotProfile(m,model=model,xaxis=xaxis,show=False,ax=ax,xmin=xmin,xmax=xmax,xL=xL,xlabel=xlabel,
									xrev=xrev,y1rev=y1rev,points=points,
									y1=y1,y1L=y1L,y1col=cm[i],
									y1label=y1label)
		elif index is not None:
			cm=[cmap(i) for i in np.linspace(0.0,0.9,len(mods))]
			for i in m.hist_dat["model_number"][index]:
				model=m.hist_dat["model_number"][index][i]
				self.plotProfile(m,model=model,xaxis=xaxis,show=False,ax=ax,xmin=xmin,xmax=xmax,xL=xL,xlabel=xlabel,
									xrev=xrev,y1rev=y1rev,points=points,
									y1=y1,y1L=y1L,y1col=cm[i],
									y1label=y1label)
		
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
		self.plotHR(m,ax=ax,maxMod=m.prof_head['model_number'],show=False)
		
		ax=plt.subplot(2,4,6)
		self.plotHistory(m,ax=ax,show=False,xaxis='log_center_T',y1='log_center_Rho',y1L='linear',
								minMod=0,maxMod=m.prof_head['model_number'],y1col='k',
								xlabel=self.labels('teff',log=True,center=True),y1label=self.labels('rho',log=True,center=True))
		
		ax=plt.subplot(1,2,2)
		self.plotAbun(m,ax=ax,show=False,xlabel=self.labels('mass'))
		
		if show==True:
			plt.show()
	    