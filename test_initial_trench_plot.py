#-------------------------------------------------------------------------------
# Name:        test_initial_trench_plot
# Purpose:
#
# Author:      Wenjiao_Wang
#
# Created:     27/03/2014
# Copyright:   (c) Wenjiao_Wang 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
from Node_class_1 import Nodes
import matplotlib.pyplot as plt

C_sticking=0.0  #assuming a constant sticking coefficient
trench=Nodes(N=150,alpha=90.0,AR=5)
length=len(trench.position)


position_array=np.array(trench.position)

plt.plot(position_array.real,position_array.imag,'r-o',trench.center_positions.real,trench.center_positions.imag,'bo')    #plot for the positions of the nodes

plt.show()
plt.plot(trench.psi_angle,'ro',trench.center_psi,'bo')
plt.show()
#plt.plot(trench.y_position,trench.psi_angle,'r-')    #plot for the psi angle for the nodes
#plt.show()

#plt.plot(trench.y_position[10:18],trench.z_position[10:18],'ro')   #plot for the trench edge for V shape trench

print trench.center_inside_trench(30)
plt.plot(position_array.real,position_array.imag,'r-',trench.center_positions[30].real,trench.center_positions[30].imag,'ro',position_array[40].real,position_array[40].imag,'ro')
plt.show()
print trench.center_connecting_slope(30,40)

direct_flux=trench.direct_flux_distribution()

plt.plot(direct_flux,'ro')
plt.show()
'''
receiving_matrix=trench.receiving_matrix()
result=np.zeros(length-1)
for i in range(0,length-1,1):
    for j in range(0,length-1,1):
        result[j]=receiving_matrix[j][i]
    plt.plot(result)

plt.show()

'''
receiving_flux_1=trench.stable_receiving_flux_vector()
plt.plot(direct_flux,'b-',receiving_flux_1,'ro')
plt.show()
