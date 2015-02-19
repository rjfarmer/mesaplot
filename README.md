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