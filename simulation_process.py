#-------------------------------------------------------------------------------
# Name:        simulation_process
# Purpose:
#
# Author:      Wenjiao_Wang
#
# Created:     27/03/2014
# Copyright:   (c) Wenjiao_Wang 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from Node_class import Nodes
import matplotlib.pyplot as plt
import numpy as np
Flux=1
Sc_surface=1.0/Flux
C_sticking=0.1  #defines a constant sticking coefficient

trench=Nodes(N=50,alpha=80.0,AR=5,Sc_surface=Sc_surface)   #Sc_surface defines the flux outside trench, which equals 1/Sc_surface
#while trench.node_optimization()==1:
#    trench.node_optimization()
length=len(trench.position)
trench_shape=np.array(trench.position)
plt.plot(trench_shape.real,trench_shape.imag,'ro')    #plot for the positions of the nodes
plt.show()

number_of_cycles=2
trench_history=[]
for i in range(0,number_of_cycles,1):
    length=len(trench.position)
    current_shape=np.zeros(length,'complex')
    for j in range(0,length,1):
        current_shape[j]=trench.position[j]   #the real part records the y indices #the imag part records the z indices
    trench_history.append(current_shape)
    receiving_flux_1=trench.stable_receiving_flux_vector_2(SC=C_sticking)

    sticking_coefficient_1=np.ones(length)*C_sticking     #trench.Sticking_coefficient_vector(receiving_flux_1)
    trench.trench_update(receiving_flux_1,sticking_coefficient_1)
current_shape=np.zeros(len(trench.position),'complex')
for j in range(0,len(trench.position),1):
    current_shape[j]=trench.position[j]   #the real part records the y indices #the imag part records the z indices
trench_history.append(current_shape)


for i in range(0,number_of_cycles+1,1):
    plt.plot(trench_history[i].real,trench_history[i].imag)

plt.show()

#receiving_flux_2=trench.stable_receiving_flux_vector()
#to the second cycle, all values become nan, need to find out why
'''
trench.y_position=trench_history[2].real
trench.z_position=trench_history[2].imag
trench.renew_psi()
'''
'''
for i in range(0,len(trench.y_position)/2,1):
    if i>0:
        trench.psi_angle[i-1]=cmath.phase(complex(trench.y_position[i]-trench.y_position[i-1],trench.z_position[i]-trench.z_position[i-1]))/np.pi*180.0+90
        trench.psi_angle[len(trench.y_position)-i]=trench.psi_angle[i-1]
trench.psi_angle[len(trench.y_position)/2-1]=cmath.phase(complex(trench.y_position[len(trench.y_position)/2]-trench.y_position[len(trench.y_position)/2-1],trench.z_position[len(trench.y_position)/2]-trench.z_position[len(trench.y_position)/2-1]))/np.pi*180.0+90
trench.psi_angle[len(trench.y_position)/2+1]=trench.psi_angle[len(trench.y_position)/2-1]
trench.psi_angle[len(trench.y_position)/2]=90.0

direct_flux_vector=trench.direct_flux_distribution_1()
sticking_coefficient_vector=trench.Sticking_coefficient_vector(direct_flux_vector)
segment_length_vector=trench.node_lengths()
emittion_flux_vector_1=trench.Emission_vector(receiving_flux_vector=direct_flux_vector,sticking_coefficient_vector=sticking_coefficient_vector,segment_length_vector=segment_length_vector) #first time re-emission

R=trench.receiving_matrix_1()   #receiving matrix

C_2=np.array(np.matrix(emittion_flux_vector_1)*np.matrix(R))+direct_flux_vector     #receiving flux vector after first emission
C_2=C_2[0]
#plt.plot(C_2[0],'ro',direct_flux_vector,'b-')

#now need to update sticking_coefficient, the emittion_flux_vector_1, and then new receiving flux vector can be computed.
sticking_coefficient_vector_2=trench.Sticking_coefficient_vector(C_2)
emittion_flux_vector_2=trench.Emission_vector(receiving_flux_vector=C_2,sticking_coefficient_vector=sticking_coefficient_vector_2,segment_length_vector=segment_length_vector) #first time re-emission
C_3=np.array(np.matrix(emittion_flux_vector_2)*np.matrix(R))+direct_flux_vector     #receiving flux vector after first emission
C_3=C_3[0]
#plt.plot(C_3,'ro',C_2,'g-',direct_flux_vector,'b-')

sticking_coefficient_vector_3=trench.Sticking_coefficient_vector(C_3)
emittion_flux_vector_3=trench.Emission_vector(receiving_flux_vector=C_3,sticking_coefficient_vector=sticking_coefficient_vector_3,segment_length_vector=segment_length_vector) #first time re-emission
C_4=np.array(np.matrix(emittion_flux_vector_3)*np.matrix(R))+direct_flux_vector     #receiving flux vector after first emission
C_4=C_4[0]
#plt.plot(C_4,'r-',C_3,'m-',C_2,'g-',direct_flux_vector,'b-')

sticking_coefficient_vector_4=trench.Sticking_coefficient_vector(C_4)

trench.trench_update(C_4, sticking_coefficient_vector_4)  #update the trench profile, and save it into a new trench instance

plt.plot(trench.y_position,trench.z_position,'ro',trench_1.real,trench_1.imag,'b-')
plt.show()

'''