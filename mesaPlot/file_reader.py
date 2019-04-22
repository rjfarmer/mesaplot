# Copyright (c) 2017, Robert Farmer r.j.farmer@uva.nl

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
import os
import bisect
import subprocess
from io import BytesIO

from distutils.version import StrictVersion

class data(object):
	def __init__(self):
		self.data={}
		self.head={}
		self._loaded=False
		self._mph=''
		self._type=''
		
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
						
		raise AttributeError("Must call "+str(self._type)+" first")
	
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
			
	def __getitem__(self,key):
		tmp=data()
		if key > np.size(self.data[self.data_names[0]]):
			raise IndexError
		elif key <0:
			x=self.data[key-1:key]
		else:
			x=self.data[key:key+1]
		
		tmp.data=np.array(x,dtype=self.data.dtype)
		tmp.head=self.head
		tmp._loaded=True
		tmp._mph=self._mph
		tmp.data_names=self.data_names
		tmp.head_names=self.head_names
		return tmp
		
	def loadFile(self, filename, max_num_lines=-1, cols=[],final_lines=-1,_dbg=False):
		if StrictVersion(np.__version__) < StrictVersion('1.10.0') or _dbg:
			f = self._loadFile1
		else:
			f = self._loadFile2
		f(filename, max_num_lines, cols, final_lines)
		
	def _loadFile1(self, filename, max_num_lines=-1, cols=[],final_lines=-1):
		numLines = self._filelines(filename)
		self.head = np.genfromtxt(filename, skip_header=1, skip_footer=numLines-4, names=True,dtype=None)
		skip_lines = 0
		if max_num_lines > 0 and max_num_lines < numLines:
			skip_lines = numLines - max_num_lines
		
		#Just the names
		names = np.genfromtxt(filename, skip_header=5, names=True, skip_footer=numLines-5,dtype=None)
		names = names.dtype.names
		
		usecols = None
		cols = list(cols)
		if len(cols):
			if ('model_number' not in cols and 'model_number' in names):
				cols = cols + ('model_number',)
			if ('zone' not in cols and 'zone' in names):
				cols = cols + ('zone',)
		
			colsSet = set(cols)
			usecols = [i for i, e in enumerate(names) if e in colsSet]
		
		if final_lines > 0:	
			line = subprocess.check_output(['tail', '-'+str(final_lines), filename])
			self.data = np.genfromtxt(BytesIO(line), names=names, usecols=usecols,dtype=None)
		else:
			self.data = np.genfromtxt(filename, skip_header=5, names=True, skip_footer=skip_lines, usecols=usecols,dtype=None)
		self.head_names = self.head.dtype.names
		self.data_names = self.data.dtype.names
		self._loaded = True
		
		
	def _loadFile2(self, filename, max_num_lines=-1, cols=[],final_lines=-1):
		# numLines = self._filelines(filename)
		self.head = np.genfromtxt(filename, skip_header=1, max_rows=1, names=True,dtype=None)
			
		#Just the names
		names = np.genfromtxt(filename, skip_header=5, names=True, max_rows=1,dtype=None)
		names = names.dtype.names
			
		usecols = None
		cols = list(cols)
		if len(cols):
			if ('model_number' not in cols and 'model_number' in names):
				cols = cols + ['model_number']
			if ('zone' not in cols and 'zone' in names):
				cols = cols + ['zone']
			
			colsSet = set(cols)
			usecols = [i for i, e in enumerate(names) if e in colsSet]
			
		if final_lines > 0:	
			line = subprocess.check_output(['tail', '-'+str(final_lines), filename])
			self.data = np.genfromtxt(BytesIO(line), names=names, usecols=usecols,dtype=None)
		else:
			if max_num_lines > 0:
				self.data = np.genfromtxt(filename, skip_header=5, names=True, max_rows = max_num_lines, usecols=usecols,dtype=None)
			else:
				self.data = np.genfromtxt(filename, skip_header=5, names=True, usecols=usecols,dtype=None)
		self.head_names = self.head.dtype.names
		self.data_names = self.data.dtype.names
		self._loaded = True

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
		self.binary=data()
		self.prof_ind=""
		self.log_fold=""
		self.clearProfCache()
		self.cache_limit=100
		self._cache_wd=''
	
		self.hist._mph='history'
		self.prof._mph='profile'
		self.hist._type='loadHistory'
		self.prof._type='loadProfile'
		
		self.hist._mph='binary'
		self.hist._mph='loadBinary'
		
	
	def loadHistory(self,f="",filename_in=None,max_model=-1,max_num_lines=-1,cols=[],final_lines=-1,_dbg=False):
		"""
		Reads a MESA history file.
		
		Optional:
		f: Folder in which history.data exists, if not present uses self.log_fold, if that is
		not set try the current working directory.
		filename_in: Reads the file given by name
		max_model: Maximum model to read into, may help when having to clean files with many retires, backups and restarts by not processing data beyond max_model
		max_num_lines: Maximum number of lines to read from the file, maps ~maximum model number but not quite (retires, backups and restarts effect this)
		cols: If none returns all columns, else if set as a list only stores those columns, will always add model_number to the list
		final_lines: Reads number of lines from end of the file if > 0
		
		
		Returns:
		self.hist.head: The header data in the history file as a structured dtype
		self.hist.data:  The data in the main body of the history file as a structured dtype
		self.hist.head_names: List of names of the header fields
		self.hist.data_names: List of names of the data fields
		
		Note it will clean the file up of backups,retries and restarts, preferring to use
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

		self.hist.loadFile(filename,max_num_lines,cols,final_lines=final_lines,_dbg=_dbg)
		
		if max_model>0:
			self.hist.data=self.hist.data[self.hist.model_number<=max_model]

		# Reverse model numbers, we want the unique elements
		# but keeping the last not the first.
		
		#Fix case where we have at end of file numbers:
		# 1 2 3 4 5 3, without this we get the extra 4 and 5
		if np.size(self.hist.model_number) > 1:
			self.hist.data=self.hist.data[self.hist.model_number<=self.hist.model_number[-1]]
			mod_rev=self.hist.model_number[::-1]
			_, mod_ind=np.unique(mod_rev,return_index=True)
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
	
		
	def loadProfile(self,f='',num=None,prof=None,mode='nearest',silent=False,cache=True,cols=[]):
		if num is None and prof is None:
			self._readProfile(f) #f is a filename
			return
		
		if len(f)==0:
			if len(self.log_fold)==0:
				self.log_fold='LOGS/'
			f=self.log_fold
		else:
			self.log_fold=f
			
		self._loadProfileIndex(f) #Assume f is a folder
		prof_nums=np.atleast_1d(self.prof_ind["profile"]).astype('int')
		
		if prof is not None:
			pos=np.where(prof_nums==prof)[0][0]
		else:
			if np.count_nonzero(self.prof_ind)==1:
				pos=0
			else:
				if num<=0:
					pos=num
				else:
				#Find profile with mode 'nearest','upper','lower','first','last'
					pos = bisect.bisect_left(self.prof_ind["model"], num)
					if pos == 0 or mode=='first':
						pos=0
					elif pos == np.size(self.prof_ind["profile"]) or mode=='last':
						pos=-1
					elif mode=='lower':
						pos=pos-1
					elif mode=='upper':
						pos=pos
					elif mode=='nearest':
						if self.prof_ind["model"][pos]-num < num-self.prof_ind["model"][pos-1]:
							pos=pos
						else:
							pos=pos-1
					else:
						raise ValueError("Invalid mode")
						
		profile_num=np.atleast_1d(self.prof_ind["profile"])[pos]		
		filename=f+"/profile"+str(int(profile_num))+".data"
		if not silent:
			print(filename)
		self._readProfile(filename,cache=cache,cols=cols)
		return
			
	def loadMod(self,filename=None):
		"""
		Fails to read a MESA .mod file.
		"""
		from io import BytesIO
		
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
			#Make a dictionary of converters 
			
		d = {k:self._fds2f for k in range(len(self.mod_dat_names))}	
			
		self.mod_dat=np.genfromtxt(filename,skip_header=count,
						names=self.mod_dat_names,skip_footer=5,dtype=None,converters=d)
		
		
	def iterateProfiles(self,f="",priority=None,rng=[-1.0,-1.0],step=1,cache=True):
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
				if type(priority) is not list: priority = [ priority ]
				if x["priority"] in priority or 0 in priority:
					self.loadProfile(f=f+"/profile"+str(int(x["profile"]))+".data",cache=cache)
					yield
			if len(rng)==2 and rng[0]>0:
				if x["model"] >=rng[0] and x["model"] <= rng[1] and np.remainder(x["model"]-rng[0],step)==0:
					self.loadProfile(f=f+"/profile"+str(int(x["profile"]))+".data",cache=cache)
					yield
				elif x["model"]>rng[1]:
					return
			elif len(rng)>2 and rng[0]>0:
				if x["model"] in rng:
					self.loadProfile(f=f+"/profile"+str(int(x["profile"]))+".data",cache=cache)
					yield
			else:
				self.loadProfile(f=f+"/profile"+str(int(x["profile"]))+".data",cache=cache)
				yield 
		return
				
	def _loadProfileIndex(self,f):
		self.prof_ind=np.genfromtxt(f+"/profiles.index",skip_header=1,names=["model","priority","profile"])

	def _readProfile(self,filename,cache=True,cols=[]):
		"""
		Reads a MESA profile file.
		
		Required:
		filename: Path to profile to read
		
		Optional:
		cache: If true caches the profile data so multiple profile loads do not need to reread the data
		cols: cols: If none returns all columns, else if set as a list only storing those columns, it will always add zone to the list of columns
		
		Returns:
		self.prof.head: The header data in the profile as a structured dtype
		self.prof.data:  The data in the main body of the profile file as a structured dtype
		self.prof.head_names: List of names of the header fields
		self.prof.data_names: List of names of the data fields
		"""
		
		# Handle cases where we change directories inside the python session
		if self._cache_wd != os.getcwd():
			self.clearProfCache()
			self._cache_wd=os.getcwd()
		
		
		if filename in self._cache_prof_name and cache:
			self.prof=self._cache_prof[self._cache_prof_name.index(filename)]
		else:
			x=data()
			x.loadFile(filename,cols=cols)
			if cache:
				if len(self._cache_prof_name)==self.cache_limit:
					self._cache_prof.pop(0)
					self._cache_prof_name.pop(0)
				self._cache_prof.append(x)
				self._cache_prof_name.append(filename)
			self.prof=x
			
	def clearProfCache(self):
		self._cache_prof=[]
		self._cache_prof_name=[]
	
	def _fds2f(self,x):
		if isinstance(x, str):
			f=np.float(x.replace('D','E'))
		else:
			f=np.float(x.decode().replace('D','E'))
		return f
		
	def abun(self,element):
		xx=0
		for ii in range(0,1000):
			try:
				xx=xx+np.sum(self.prof.data[element+str(ii)]*10**self.prof.logdq)
			except:
				pass
		return xx

	def loadBinary(self,f="",filename_in=None,max_model=-1,max_num_lines=-1,cols=[]):
		"""
		Reads a MESA binary history file.
		
		Optional:
		f: Folder in which binary_history.data exists, if not present uses self.log_fold, if that is
		not set try the current working directory.
		filename_in: Reads the file given by name
		max_model: Maximum model to read into, may help when having to clean files with many retries, backups and restarts by not processing data beyond max_model
		max_num_lines: Maximum number of lines to read from the file, maps ~maximum model number but not quite (retries, backups and restarts effect this)
		cols: If none returns all columns, else if set as a list only stores those columns, it will always add model_number to the list
		
		
		Returns:
		self.binary.head: The header data in the history file as a structured dtype
		self.binary.data:  The data in the main body of the history file as a structured dtype
		self.binary.head_names: List of names of the header fields
		self.binary.data_names: List of names of the data fields
		
		Note it will clean the file up of backups, retries and restarts, preferring to use
		the newest data line.
		"""
		if len(f)==0:
			if len(self.log_fold)==0:
				self.log_fold='./'
			f=self.log_fold
		else:
			self.log_fold=f+"/"

		if filename_in is None:               
			filename=os.path.join(self.log_fold,'binary_history.data')
		else:
			filename=filename_in

		self.binary.loadFile(filename,max_num_lines,cols)
		
		if max_model>0:
			self.binary.data=self.binary.data[self.binary.model_number<=max_model]

		# Reverse model numbers, we want the unique elements
		# but keeping the last not the first.
		
		#Fix case where we have at end of file numbers:
		# 1 2 3 4 5 3, without this we get the extra 4 and 5
		self.binary.data=self.binary.data[self.binary.model_number<=self.binary.model_number[-1]]
		
		mod_rev=self.binary.model_number[::-1]
		_, mod_ind=np.unique(mod_rev,return_index=True)
		self.binary.data=self.binary.data[np.size(self.binary.model_number)-mod_ind-1]


class inlist(object):
    def __init__(self):
        pass
        
    def read(self,filename):
        res = {}
        with open(filename,'r') as f:
            for l in f:
                l=l.strip()
                if l.startswith('!') or not len(l.strip()):
                    continue
                if '=' in l:
                    line = l.split('=')
                    res[line[0].strip()] = line[1].strip()
        return res
                
        
