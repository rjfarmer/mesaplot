import mesaPlot as mp
import matplotlib.pyplot as plt


m=mp.MESA()
m.loadHistory()
m.loadProfile(num=-1)
p=mp.plot()

#Simply plot
p.plotAbun(m)


#Advanced
fig=plt.figure(figsize=(10,8))
ax=fig.add_subplot(111)
p.plotAbun(m,num_labels=6,abun_random=True,yrng=[-6.5,0.5],show_title_model=True,show_title_age=True,fig=fig,ax=ax)
