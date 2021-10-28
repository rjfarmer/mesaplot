

[![DOI](https://zenodo.org/badge/30720868.svg)](https://zenodo.org/badge/latestdoi/30720868)


# mesaplot
Library of python routines to read MESA ouput files and plot MESA quantites

## Installation instructions:
git clone the repo then call
````bash
python3 setup.py install 
````

Depending on choice of python version, --user can also be passed to install locally

````bash
make
````

Can be called as well

## Testing

````bash
pytest
````

To run tests for current python version

````tox
tox
````

will run tests for several python versions

## Contributing

Bug reports should go to githubs issue tracker.

Requests for new features should also got to the issue tracker.

If you wish to submit your own fix/new feature please send a pull request.


## Reading data:

````python
import mesaPlot as mp
m=mp.MESA()
````

Now m contains all the useful stuff

### History files
````python
m.loadHistory()
````
This loads up the history file data by default it will look for LOGS/history.data.
But if you have a different folder to look at then you can either

````python
m.log_fold='new_folder/LOGS/'
m.loadHistory()
````
or
````python
m.loadHistory(f='new_folder/LOGS/')
````
Note this will automatically clean the history data of retries, backups and restarts. To write that data back to disk
````python
m.scrubHistory()
````
Which will create a file "LOGS/history.data.scrubbed" if you don't want that then:
````python
m.scrubHistory(fileOut='newFile')
````

Data can be accessed as m.hist.data['COLUMN_NAME'] or m.hist.COLUMN_NAME. The second option is
also tab completable. The header information is either m.hist.head['COLUMN_NAME'] or m.hist.COLUMN_NAME.

### Profile files
To load a profile file its:
````python
m.loadProfile()
````
Again you change the LOGS folder either with log_fold or f=.
To choose which file to load, either:
````python
m.loadProfile(num=MODEL_NUMBER)
or
m.loadProfile(prof=PROFILE_NUMBER)
````
Where MODEL_NUMBER is a MESA model number and PROFILE_NUMBER is the number in the profile file name.
You can also set a mode
````python
m.loadProfile(num=MODEL_NUMBER,mode='first|lower|upper|nearest')
````
This is for when the model you want isn't in the data. Either we load the first model, the model just before the one you want, the model just after the one you want or the nearest (above or below) the model you want.
There are also two special model numbers 0 for first model and a negative number that counts backwards (-1 is the last model, -2 is last but one etc)

Data can be accessed as m.prof.data['COLUMN_NAME'] or m.prof.COLUMN_NAME. The second option is
also tab completable. The header information is either m.prof.head['COLUMN_NAME'] or m.prof.COLUMN_NAME.


## Plotting

Here we'll show the basics of plotting, there are more complex examples for each section. Commands will assume you are in a MESA work folder, such that the data is in a LOGS/ folder.


### History data

````python
import mesaPlot as mp
m=mp.MESA()
p=mp.plot()
m.loadHistory()
p.plotHistory(m,xaxis='star_age',y1='log_center_T',y2='he_core_mass')
````

### Profile data

````python
import mesaPlot as mp
m=mp.MESA()
p=mp.plot()
m.loadProfile(num=-1)
p.plotProfile(m,xaxis='mass',y1='logT',y2='ye')
````

### HR

````python
import mesaPlot as mp
m=mp.MESA()
p=mp.plot()
m.loadHistory()
p.plotHR(m)
````

### Kippenhan's

Kippenhan plot with model number vs mass. Note all Kippenhan plots
require your history file to have both mixing_regions X and burning_regions Y
set in your history_columns.list file, where X and Y are integers that 
specify the maximum number of mixing/burning
zones in your model, values around 20 will usually suffice. Models going to
core collapse may want to increase this limit to 40.


````python
import mesaPlot as mp
m=mp.MESA()
p=mp.plot()
m.loadHistory()
p.plotKip(m,show_mass_loc=True)
````

Kippenhan plot with star age vs mass
````python
import mesaPlot as mp
m=mp.MESA()
p=mp.plot()
m.loadHistory()
p.plotKip2(m)
````

Generic kippenhan plotter
````python
import mesaPlot as mp
m=mp.MESA()
p=mp.plot()
m.loadHistory()
p.plotKip3(m,show_mass_loc=True)
````
![Kippenhan plotKip3 SAGB star](/examples/kip.png?raw=true "Kippenhan plot")

New way of doing plotKip2 (star age vs mass)
````python
import mesaPlot as mp
m=mp.MESA()
p=mp.plot()
m.loadHistory()
p.plotKip3(m,xaxis='star_age',age_lookback=True,age_log=True)
````
![Kippenhan plotKip3 SAGB star 2](/examples/kip_age.png?raw=true "Kippenhan plot 2")

Profile based kippenhans
````python
import mesaPlot as mp

m=mp.MESA()
m.loadHistory()
m.loadProfile(num=1)
p=mp.plot()
p.plotKip3(m,plot_type='profile',xaxis='model_number',yaxis='mass',zaxis='logRho',mod_min=1,mod_max=3000)
````
![Kippenhan plotKip3 SAGB star 3](/examples/kip_prof.png?raw=true "Kippenhan plot 3")


### Abundances

````python
import mesaPlot as mp
m=mp.MESA()
p=mp.plot()
m.loadProfile(num=-1)
p.plotAbun(m)
````

![Basic abundance plot](/examples/abun_basic.png?raw=true "Abundance plot")

````python
import mesaPlot as mp
m=mp.MESA()
p=mp.plot()
m.loadProfile(num=-1)
p.plotAbunByA(m)
````

![Production plot](/examples/abun_bya.png?raw=true "Production plot")

````python
import mesaPlot as mp
m=mp.MESA()
p=mp.plot()
m.loadProfile(num=-1)
m2=mp.MESA()
m2.log_fold='../some/other/mesa/result'
m2.loadprofile(num=-1)
p.plotAbunByA(m,m2=m2)
````

````python
import mesaPlot as mp
m=mp.MESA()
p=mp.plot()
p.set_solar('ag89')
m.loadProfile(num=-1)
#Plot the mass fractions relative to solar
p.plotAbunByA(m,stable=True)
````

````python
import mesaPlot as mp
m=mp.MESA()
p=mp.plot()
p.set_solar('ag89')
m.loadProfile(num=-1)
m2=mp.MESA()
m2.log_fold='../some/other/mesa/result'
m2.loadprofile(num=-1)
#Plot the mass fractions relative to solar relative to 2nd model
p.plotAbunByA(m,m2=m2,stable=True)
````

````python
import mesaPlot as mp
m=mp.MESA()
p=mp.plot()
m.loadHistory()
#Use the data in the history file at model_number==model
p.plotAbunByA(m,plot_type='history',model=1000,prefix='log_center_')
````


````python
import mesaPlot as mp
m=mp.MESA()
p=mp.plot()
p.set_solar('ag89')
m.loadHistory()
#Use the data in the history file, plotting relative to another model number after decaying the isotopes to thier stable versions
p.plotAbunByA(m,plot_type='history',model=1000,model2=1,prefix='surface_',stable=True)
````


````python
import mesaPlot as mp
m=mp.MESA()
p=mp.plot()
m.loadProfile(num=-1)
p.plotAbunPAndN(m)
````

![Nuclear abundances plot](/examples/abun_bypandn.png?raw=true "Nuclear chart")

````python
import mesaPlot as mp
m=mp.MESA()
p=mp.plot()
m.loadProfile(num=-1)
p.plotAbunPAndN(m,plot_type='history',model=1000,prefix='log_center_')
````

````python
import mesaPlot as mp
m=mp.MESA()
p=mp.plot()
m.loadHistory()
p.plotAbunHist(m)
````

### Burn data

````python
import mesaPlot as mp
m=mp.MESA()
p=mp.plot()
m.loadProfile(num=-1)
p.plotBurn(m)
````

````python
import mesaPlot as mp
m=mp.MESA()
p=mp.plot()
m.loadHistory()
p.plotBurnHist(m)
````

### Dynamos

````python
import mesaPlot as mp
m=mp.MESA()
p=mp.plot()
m.loadProfile(num=-1)
p.plotDynamo(m)
````

![Dynamo  50M_z2m2_high_rotation](/examples/dynamo1.png?raw=true "Dynamo 1 plot")

````python
import mesaPlot as mp
m=mp.MESA()
p=mp.plot()
m.loadProfile(num=-1)
p.plotDyanmo2(m)
````

![Dyanmo 2 50M_z2m2_high_rotation](/examples/dynamo2.png?raw=true "Dynamo 2 plot")

### Angular momentum mixing

````python
import mesaPlot as mp
m=mp.MESA()
p=mp.plot()
m.loadProfile(num=-1)
p.plotAngMom(m)
````

### Time series Profile plots
````python
import mesaPlot as mp

m=mp.MESA()
m.loadHistory()
m.loadProfile(num=1)
p=mp.plot()
p.plotSliderProf(m,'plotAbun')
````

Passing in a string for the name of a plotting function, (only ones based on profile data).
This will show that plot with a slider that can be used to iterate over the available profile files
plotSliderProf will take any extra arguments passed to it and
pass them to the plotting function.


### Stacked plots

### Multi profile plots

### Grid plotting


