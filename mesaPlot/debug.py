# Copyright (c) 2017, Robert Farmer rjfarmer@asu.edu

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
from matplotlib.ticker import MaxNLocator,AutoMinorLocator
import matplotlib.patheffects as path_effects
import os
import glob
import subprocess

	
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
