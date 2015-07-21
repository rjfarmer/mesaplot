import mesaPlot as mp
import matplotlib.pyplot as plt


m=mp.MESA()
m.loadHistory()
p=mp.plot()

fig=plt.figure(figsize=(12,10))
ax=fig.add_subplot(1)

#Simply plot
p.plotKip(m,fig=fig,ax=ax,show=False)
plt.show()

