#-------------------------------------------------------------------------------
# Name:        test_direct_flux_distribution
# Purpose:
#
# Author:      Wenjiao_Wang
#
# Created:     05/03/2014
# Copyright:   (c) Wenjiao_Wang 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from initialization_of_trench_shape import Nodes
import numpy as np
import matplotlib.pyplot as plt
import time

d1=100.0
N1=50
alpha1=90.0
AR1=5.0
trench=Nodes(d=d1,N=N1,alpha=alpha1,AR=AR1,K_number=10,Sc_surface=1)


time0=time.time()
direct_flux=trench.direct_flux_distribution()
time1=time.time()
time_total=time1-time0  #the time spent on calculating the direct flux for the nodes
plt.plot(trench.z_position,direct_flux,'ro')
plt.show()
print time_total
