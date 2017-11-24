import mesaPlot as mp
import matplotlib.pyplot as plt


m=mp.MESA()
m.log_fold='examples/LOGS/'

m.loadHistory()
m.loadProfile(num=-1)
p=mp.plot()

fig=plt.figure(figsize=(12,10))
ax=fig.add_subplot(111)
p.plotDynamo(m,y1rng=[0,10],fig=fig,ax=ax,show_rotation=False)

fig=plt.figure(figsize=(12,10))
ax=fig.add_subplot(111)
p.plotDynamo(m,y1rng=[0,10],fig=fig,ax=ax,show_rotation=True)
