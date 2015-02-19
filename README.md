# mesaplot
Library of python routines to read MESA ouput files and plot MESA quantites


How to use:

Reading data:

````python
import mesaPlot as mp
m=mp.MESA()
````

Now contains all the usefull stuff

````python
m.loadHistory()
````
This loads up the history file data by default it will look for LOGS/history.data.
But if you have a different folder to look at then you can either

````python
m.log_fold='new_folder'
````
or
````python
m.loadHistory(f='new_folder')
````
Note this will automatically clean the history data of retries,backups and restarts. To write that data back to disk 
````python
m.scrubHistory()
````
Which will create a file "LOGS/history.data.scrubbed" if you dont want that then:
````python
m.scrubHistory(fileOut='newFile')
````

To load aa profile file then its:
````python
m.loadProfile()
````
Again you change the LOGS folder eeither with log_fold or f=.
To choose which file to load eithe:
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

Plotting