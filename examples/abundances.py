import mesaPlot as mp
import matplotlib.pyplot as plt


m=mp.MESA()

m.log_fold='examples/LOGS/'

m.loadHistory()
m.loadProfile(num=-1)
p=mp.plot()

#Simply plot
p.plotAbun(m)

#Advanced
fig=plt.figure(figsize=(12,12))
ax=fig.add_subplot(111)
p.plotAbun(m,num_labels=6,rand_col=True,y1rng=[-6.5,0.5],show_title_model=True,show_title_age=True,fig=fig,ax=ax,
xmax=5.0)


# Abundace as function of atomic mass
fig=plt.figure(figsize=(12,12))
ax=fig.add_subplot(111)
p.plotAbunByA(m,mass_range=[0.0,5.0],fig=fig,ax=ax)

# Abundance as a function of proton and neutron number
fig=plt.figure(figsize=(12,12))
ax=fig.add_subplot(111)
p.plotAbunPAndN(m,mass_range=[0.0,5.0],mass_frac_rng=[10**-5,1.0],fig=fig,ax=ax)
