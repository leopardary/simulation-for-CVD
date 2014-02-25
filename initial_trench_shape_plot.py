from initialization_of_trench_shape import Nodes
import matplotlib.pyplot as plt

trench=Nodes()
trench.initial_shape(d=100.0,N=100,alpha=50.0,AR=0.5)
plt.plot(trench.x_position,trench.z_position,'r-')    #plot for the positions of the nodes
plt.show()
plt.plot(trench.x_position,trench.psi_angle,'r-')    #plot for the psi angle for the nodes
plt.show()