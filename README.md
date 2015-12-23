# mesaplot
Library of python routines to read MESA ouput files and plot MESA quantites

## Installation instructions:
Simply git clone the repo then add the folder to your PYTHONPATH
````bash
export PYTHONPATH=$PYTHONPATH:/path/to/mesaplot/folder
````

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
also tab completable. The header information is either m.hist.head['COL_NAME'] or m.hist.COL_NAME.

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
This is for when the model you want isn't in the data. Either we load the first model, the model just before the one you, the model just after the one you want or the nearest (above or below) the model you want.
There are also two special model numbers 0 for first model we have and -1 for the last.

Data can be accessed as m.prof.data['COLUMN_NAME'] or m.prof.COLUMN_NAME. The second option is 
also tab completable. The header information is either m.prof.head['COL_NAME'] or m.prof.COL_NAME.

### Mod files
To load a mod file its:
````python
m.loadMod(FILENAME)
```` 


## Plotting

### Intro
Generally the plotting routines follow this structure:
````python
p=mp.plot()
m.loadHistory()|m.loadProfile(num=NUM)
p.plotSomething(m)
````
So we want to make sure we load either the history or the profile data before calling the plot function and the only required argument
is passing an instance of the mp.MESA() class with the data loaded.

Depending on the function some of these may not apply:
````python
xaxis='model_number' #Column name from history|profile file
y1='star_mass' #Column name from history|profile file
y2=None #Column name from history|profile file, adds line to right hand axis
show=True #Should we immediately show the plot or wait
ax=None #A axis instance, useful for grid plotting, see later on
fig=None # Pass a fig instance
xmin=None #min x value to show
xmax=None #max x value to show
xlog=False #Whether axis should be linear or log10 (if the axis is already a log quantity then leave as linear)
y1log=False #Whether axis should be linear or log10 (if the axis is already a log quantity then leave as linear)
y2log=False #Whether axis should be linear or log10 (if the axis is already a log quantity then leave as linear)
y1col='b' #Colour of line (as well as the axis label)
y2col='r' #Colour of line (as well as the axis label)
minMod=0 #For history plots minimum model number to show
maxMod=-1 #For history plots maximum model number to show -1 means the last element in the dataset
xrev=False #Whether to reverse the xaxis
y1rev=False #Whether to reverse the y1axis
y2rev=False #Whether to reverse the y2axis
points=False #Whether to add a coloured dot at each data point (For each model number (history) or zone (profile))
xlabel=None # axis label, if None we attempt to get guess from the xaxis (see the labels function) other wise we show the name of xaxis
y1label=None # axis label, if None we attempt to get guess from the y1 (see the labels function) other wise we show the name of y1
y2label=None # axis label, if None we attempt to get guess from the y2 (see the labels function) other wise we show the name of y2
cmap=plt.cm.gist_ncar #when plotting mutplie lines what colourmap should we cycle through
yrng=[0.0,10.0] # Min and max vlues of the y axis
show_burn=False #Show regions of nuclear burning
show_burn_x=False
show_burn_line=False #Whether to show burn regions on the line (show_burn_line) or at the bottom along the xaxis (show_burn_x)
show_burn_2=False #Show regions of nuclear burning on the second yaxis
show_mix=False #Show regions of  mixing
show_mix_x=False
show_mix_line=False #Whether to show mix regions on the line (show_mix_x) or at the bottom along the xaxis (show_mix_x)
show_mix_2=False #Show regions of  mixing on the second yaxis
fx=None
fy,fy1,fy2=None #Accepts a lamda/function to transform the data either xaxis (fx) or y axis(fy) before plotting
show_title=False #Whether to show the plot title
show_title_model=False #Whether to show the model number in topt right corner
show_title_age=False #Whether to show the star age in top left corner
````

Now some remarks about each plot and any extra options available for that plot:

### Abundance plots
Plots the abundances from a profile file:
````python
plotAbun()
````
````python
num_labels=3 #Number of labels to show on the line
abun=None #A lits of isotopes to show, if None shows all available in the profile file
abun_random=False #Randomizes the colourmap, so isotopes that are near each other in the profile file, dont end up with similar colours
````
![Abundance plot 20M_si_burn](/examples/abundances.png?raw=true "Abundance plot")

### Dynamo's
Plots the B fields from a profile file:
````python
plotDynamo()
````
![Dynamo  50M_z2m2_high_rotation](/examples/dynamo1.png?raw=true "Dynamo 1 plot")

Show the rotation terms terms as well as magnetic fields
````
plotDynamo2()
````
![Dyanmo 2 50M_z2m2_high_rotation](/examples/dynamo2.png?raw=true "Dynamo 2 plot")

### Angular momentum mixing
Plots the am_log_D_* terms from a profile file:
````python
plotAngMom()
````

### Burn data
Plots the energy generated per reaction (burn_ fields plus "pp","cno","tri_alfa","c12_c12","c12_o16","o16_o16","pnhe4","photo","other" terms) from a profile file:
````python
plotBurn()
````
````python
num_labels=3 #Number of labels to show on the line
````

### General hiistory|profile plotting
General plotting routines for profile|history data
````python
plotProfile()
plotHistory()
````

Profile plots can accept
````python
mod=NUM
````
To load model number NUM

### Kippenhan 
Plots a Kippenhan diagram for the star from the history data, requires the history data has mixing_regions and burn_regions. Mixing regions shown with same colours as MESA.
````python
plotKip()
````
````python
reloadHistory=False #Whether to reload the history file
xaxis='num' #Can only be num for model number at the moment.
ageZero=0.0 #Ignore
xrng=[-1,-1] #Range of models to plot, plot cna be slow so if you dont need them dont plot them
mix=None #Mixing regions to show, None shows all, -1 means no mixing, otherwise a list of the integers (see MESA) of the mixing types ot show
cmin=None #Min value of burn to show
cmax=None #max value of burn power to show. Note will be made symmetric with max(abs(cmin),abs(cmax))
burnCmap=[cm.Purples_r,cm.hot_r] #Creates a diverging colour map for the burn data, with the first cmap used for <0 and second for >0
````
![Kippenhan plot 1M_pre_ms_to_wd](/examples/kip.png?raw=true "Kippenhan plot")


### log Rho log Teff
Plots log rho-logTeff
````python
plotTRho()
````
![Log Rho vs log T 1M_pre_ms_to_wd](/examples/trho.png?raw=true "Log Rho vs log T with burn and mix data")

### HR plot
Plots HR diagram
````python
plotHR()
````

### Stacked plots
Plots multiple profile|history plots with same xaxis, and removes the gap between the plots
````python
stackedPlots()
````
````python
typ='profile' #History or profile data
num=1 #Number of plots to show num>=2
y1rev=[]
y2rev=[]
y1=[]
y2=[]
y1L=[]
y2L=[]
y1col=[]
y2col=[]
y1label=[]
y2label=[] #same as for plotProfile and plotHistory, except as a list, starting from the top plot downwards. If not left empty, must specify for ech plot (even if you just insert a None)
````

### Multi profiles
Plots multiple profiles on one plot
````python
plotMultiProfiles() 
````
````python
mods=None #Either set mods or index, mods must be list of model_numbers
index=None # A index on the history data, ie ind=(m.hist_dat["logT"]>3.5)&(m.hist_dat["logT"]>4.0)
````

### Grid plotting
Plots the plotTRho, plotHR, history plot and abundance plots on one plot. Demonstrates how to make  your own grid plots.
````python
plotGrid2()
````
Note stackPlots() can't currently be added to a grid plot
	
	



