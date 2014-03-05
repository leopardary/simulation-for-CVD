#-------------------------------------------------------------------------------
# Name:        plot_for_direct_flux_distribution
# Purpose:
#
# Author:      Wenjiao_Wang
#
# Created:     04/03/2014
# Copyright:   (c) Wenjiao_Wang 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from initialization_of_trench_shape import Nodes
import numpy as np
import matplotlib.pyplot as plt

d1=100.0
N1=100
alpha1=90.0
AR1=5.0
trench=Nodes(d=d1,N=N1,alpha=alpha1,AR=AR1,K_number=10,Sc_surface=1)

direct_flux=[]
index=0
direct_z=[]
while (index<len(trench.z_position)/2):
    temp=trench.integrate_direct_flux(node_index=index)
    direct_flux.extend([temp[0]])
    direct_z.extend([trench.z_position[index]])
    index+=20

plt.plot(direct_z,direct_flux,'ro')
plt.show()
