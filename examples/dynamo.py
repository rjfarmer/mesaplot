import mesaPlot as mp
import matplotlib.pyplot as plt


m=mp.MESA()
m.loadHistory()
m.loadProfile(num=-1)
p=mp.plot()

fig=plt.figure(figsize=(10,8))
ax=fig.add_subplot(111)

p.plotDynamo(m,yrng=[0,10],fig=fig,ax=ax)

fig=plt.figure(figsize=(10,8))
ax=fig.add_subplot(111)
p.plotDynamo2(m,y1rng=[0,10],fig=fig,ax=ax)
