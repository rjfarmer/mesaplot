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
		ind=np.zeros(np.size(self.hist_dat["model_number"]))
		rev=np.copy(self.hist_dat["model_number"][::-1])
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
				raise "Invalid mode"
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
					self.mod_dat[i][j]=float(str(self.mod_dat[i][j]).replace('D','E').replace('b','').replace('\'',''))

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


class plot():
	def plotAbun(self,m,model=None,show=True,ax=None,xaxis='mass',xmin=None,xmax=None,yrng=[-3.0,1.0],cmap=plt.cm.gist_ncar,num_labels=3):
		if ax ==None:
			fig=plt.figure()
			ax=fig.add_subplot(111)
		#m.loadProfile(num=int(model))
		
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

		num_plots=0
		for i in m.prof_dat.dtype.names:
			if len(i)<=4 and len(i)>=2:
				if i[0].isalpha() and (i[1].isalpha() or i[1].isdigit()) and any(char.isdigit() for char in i):
					if np.count_nonzero((np.log10(m.prof_dat[i][mInd])>=yrng[0])&(np.log10(m.prof_dat[i][mInd])<=yrng[1])):
						num_plots=num_plots+1
			
		try:
			plt.gca().set_color_cycle([cmap(i) for i in np.linspace(0.0,0.9,num_plots)])
		except:
			pass
		
		for i in m.prof_dat.dtype.names:
			if len(i)<=4 and len(i)>=2:
				if i[0].isalpha() and (i[1].isalpha() or i[1].isdigit()) and any(char.isdigit() for char in i):
					try:
						if np.count_nonzero((np.log10(m.prof_dat[i][mInd])>=yrng[0])&(np.log10(m.prof_dat[i][mInd])<=yrng[1])):
							line, =ax.plot(m.prof_dat[xaxis][mInd],np.log10(m.prof_dat[i][mInd]),label=i,linewidth=2)
							
							for ii in range(1,num_labels+1):
								f = interpolate.interp1d(m.prof_dat[xaxis][mInd],np.log10(m.prof_dat[i][mInd]))
								xp1=((xrngL[1]-xrngL[0])*(ii/(num_labels+1.0)))+xrngL[0]
								yp1=f(xp1)
								ax.annotate(i, xy=(xp1,yp1), xytext=(xp1,yp1),color=line.get_color(),fontsize=12)
					
					except:
						pass

		#ax.legend(loc=0,fontsize=20)
		ax.set_xlabel(xaxis)
		ax.set_ylabel(r'$\log_{10}$ Abundance')
		ax.set_ylim(yrng)
		ax.set_xlim(xrngL)
		ax.set_title("Abundance")
		if show:
			plt.show()

	def plotDynamo(self,m,model=1,show=True,ax=None,xmin=None,xmax=None):
		if ax ==None:
			fig=plt.figure()
			ax=fig.add_subplot(111)
	
		m.loadProfile(num=int(model))
		if model is not None:
			m.loadProfile(num=int(model))
		mInd=np.zeros(np.size(m.prof_dat["mass"]),dtype='bool')
		mInd[:]=True
		xrngL=[m.prof_dat["mass"].min(),m.prof_dat["mass"].max()]
		if xmin is not None:
			mInd=(m.prof_dat[x]>=xmin)
			xrngL[0]=xmin

		if xmax is not None:
			mInd=mInd&(m.prof_dat[x]<=xmax)
			xrngL[1]=xmax
		
		ind=mInd&(m.prof_dat['dynamo_log_B_r']>-90)
		ax.plot(m.prof_dat["mass"][ind],m.prof_dat['dynamo_log_B_r'][ind],label='B_r',linewidth=2)
		ind=mInd&(m.prof_dat['dynamo_log_B_phi']>-90)
		ax.plot(m.prof_dat["mass"][ind],m.prof_dat['dynamo_log_B_phi'][ind],label='B_phi',linewidth=2)

		try:
			ax.legend(loc=0)
		except:
			pass
		#ax.set_ylim(0.0,10.0)
		
		ax.set_xlim(xrngL)
		ax.set_title("Dynamo Model= "+str(int(m.prof_head["model_number"]))+" Target= "+str(int(model)))
		if show:
			plt.show()

	def plotAngMom(self,m,model=1,show=True,ax=None,xmin=None,xmax=None):
		if ax ==None:
			fig=plt.figure()
			ax=fig.add_subplot(111)
			
		if model is not None:
			m.loadProfile(num=int(model))
		mInd=np.zeros(np.size(m.prof_dat["mass"]),dtype='bool')
		mInd[:]=True
		xrngL=[m.prof_dat["mass"].min(),m.prof_dat["mass"].max()]
		if xmin is not None:
			mInd=(m.prof_dat[x]>=xmin)
			xrngL[0]=xmin

		if xmax is not None:
			mInd=mInd&(m.prof_dat[x]<=xmax)
			xrngL[1]=xmax


		for i in m.prof_dat.dtype.names:
			if "am_log" in i:
				try:
					ind=mInd&m.prof_dat[i]>-90.0
					ax.plot(m.prof_dat["mass"][ind],m.prof_dat[i][ind],label=i)
				except:
					pass

		try:
			ax.legend(loc=0)
		except:
			pass

		ax.set_xlim(xrngL)
		ax.set_title("Ang Mom Model= "+str(int(m.prof_head["model_number"]))+" Target= "+str(int(model)))
		if show:
			plt.show()


	def plotProfile(self,m,model=None,x='mass',y1='',y2='',show=True,ax=None,xmin=None,xmax=None,y1L='log',y2L='log',y1col='b',y2col='r'):
		if ax ==None:
			fig=plt.figure()
			ax=fig.add_subplot(111)
			
		if model is not None:
			m.loadProfile(num=int(model))
		mInd=np.zeros(np.size(m.prof_dat["mass"]),dtype='bool')
		mInd[:]=True
		xrngL=[m.prof_dat[x].min(),m.prof_dat[x].max()]
		if xmin is not None:
			mInd=(m.prof_dat[x]>=xmin)
			xrngL[0]=xmin

		if xmax is not None:
			mInd=mInd&(m.prof_dat[x]<=xmax)
			xrngL[1]=xmax

		if len(y1)>0:
			try:
				if y1L=='log':
					ax.plot(m.prof_dat[x][mInd],np.log10(m.prof_dat[y1][mInd]),c=y1col,linewidth=2)
				else:
					ax.plot(m.prof_dat[x][mInd],m.prof_dat[y1][mInd],c=y1col,linewidth=2)
				ax.set_ylabel(y1, color=y1col)
			except:
				pass

		if len(y2)>0:
			try:
				ax2 = ax.twinx()
				if y2L=='log':
					ax.plot(m.prof_dat[x][mInd],np.log10(m.prof_dat[y2][mInd]),c=y2col,linewidth=2)
				else:
					ax.plot(m.prof_dat[x][mInd],m.prof_dat[y2][mInd],c=y2col,linewidth=2)
				ax2.set_ylabel(y2, color=y2col)
			except:
				pass

		ax.set_xlim(xrngL)
		ax.set_title("Profile Model= "+str(int(m.prof_head["model_number"]))+" Target= "+str(int(model)))
		if show:
			plt.show()
			

	def kip(self,m,show=True,reloadHistory=False,xaxis='num',ageZero=0.0,ax=None,xrng=[-1,-1],mix=None,cmin=None,cmax=None):
		if ax ==None:
			fig=plt.figure()
			ax=fig.add_subplot(111)
		
		cmap = mpl.colors.ListedColormap([[0.18, 0.545, 0.34], [0.53, 0.808, 0.98],
			[0.96, 0.96, 0.864], [0.44, 0.5, 0.565],[0.8, 0.6, 1.0],
			[0.0, 0.4, 1.0],[1.0, 0.498, 0.312],[0.824, 0.705, 0.55]])

		cmap.set_over((1., 1., 1.))
		cmap.set_under((0., 0., 0.))
		#bounds = [-0.01,0.99,1.99,2.99,3.99,4.99,5.99,6.99,7.99,8.99]
		bounds=[-1,0,1,2,3,4,5,6,7,8,9]
		norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

		if reloadHistory:
			m.loadHistory()
		
		if xrng[0]>=0:
			modInd=(m.hist_dat["model_number"]>=xrng[0])&(m.hist_dat["model_number"]<=xrng[1])
		else:	
			modInd=np.zeros(np.size(m.hist_dat["model_number"]),dtype='bool')
			modInd[:]=True
			
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

		
		newCm=self.mergeCmaps([cm.Purples_r,cm.hot_r],[[0.0,0.5],[0.5,1.0]])
		
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
		ax.imshow(mixZones.T,cmap=cmap,norm=norm,extent=extent,interpolation='nearest',origin='lower',aspect='auto')
		#ax.contourf(XX,YY,mixZones.T,cmap=cmap,norm=norm,origin='lower')
		mixZones=0
		plt.xlabel(r"$\rm{Model\; number}$")
		plt.ylabel(r"$\rm{Mass}\; [M_{\odot}]$")
		
		cb=plt.colorbar(im1)
		cb.solids.set_edgecolor("face")

		cb.set_label(r'$\rm{sign}\left(\epsilon_{\rm{nuc}}-\epsilon_{\nu}\right)\log_{10}\left(\rm{max}\left(1.0,|\epsilon_{\rm{nuc}}-\epsilon_{\nu}|\right)\right)$')
		fig.set_size_inches(12,9.45)
		
		#ax.locator_params(nbins=6)
		ax.yaxis.set_major_locator(MaxNLocator(6))
		ax.xaxis.set_major_locator(MaxNLocator(6))
		
		ax.yaxis.set_minor_locator(AutoMinorLocator(5))
		ax.xaxis.set_minor_locator(AutoMinorLocator(5))
		#ax.set_tick_params(axis='both',which='both')
		
		if show:
			plt.show()
		
	def plotTRho(self,m,model=1,show=True,ax=None,massrng=[-1,-1],l=False):
		if ax ==None:
			fig=plt.figure()
			ax=fig.add_subplot(111)

		m.loadProfile(num=int(model))
		if massrng[0] >0:
			ind=(m.prof_dat["mass"]>=massrng[0])&(m.prof_dat["mass"]<=massrng[1])
		else:
			ind=np.zeros(np.size(m.prof_dat["mass"]),dtype='bool')
			ind[:]=True

		lab=''
		if l==True:
			lab=str(int(m.prof_head["model_number"]))
		ax.plot(np.log10(m.prof_dat["temperature"][ind]),m.prof_dat['logRho'][ind],c='r',label=lab)

		#ax.legend(loc=0)
		ax.set_title("Profile Model= "+str(int(m.prof_head["model_number"]))+"Target= "+str(int(model)))
		ax.set_xlabel('Log T')
		ax.set_ylabel('Log Rho')

		ax.set_xlim(np.log10(m.prof_dat["temperature"][ind].min()),np.log10(m.prof_dat["temperature"][ind].max()))
		ax.set_ylim(m.prof_dat['logRho'][ind].min(),m.prof_dat['logRho'][ind].max())
		ax.set_title("Temp-Rho Model= "+str(int(m.prof_head["model_number"]))+"Target= "+str(int(model)))

		if show==True:
			plt.show()

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
		

		

	    