import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class Nodes:
    def __init__(self):   
        self.x_position = []
        self.z_position = []
        self.psi_angle = []
    
    def initial_shape(self, d=100.0, N=100, alpha=90.0, AR=np.inf):    #building the initial nodes for the bare trench
        self.x_position.extend(np.linspace(-d,-d/2,N,endpoint=True))
        self.z_position.extend(np.zeros(N))
        self.psi_angle.extend(np.ones(N)*90)
        step = d/2/(N-1)
        if np.tan(alpha/180*np.pi)<2*AR:    #the AR given is too large, and the trench is a 'V' shape, no flat part is included at the bottom
            L = d/2/np.cos(alpha/180*np.pi)    #L is the length of the trench wall
            N_L=int(L/step)    #N_L is the number of points on the trench wall, while keeping relatively the same point density
            self.x_position.extend(np.linspace(-d/2,0,N_L,endpoint=False))
            self.z_position.extend(np.linspace(0,-d/2*np.tan(alpha/180*np.pi),N_L,endpoint=False))
            self.psi_angle.extend(np.ones(N_L)*(90-alpha))
            self.x_position.extend(np.linspace(0,d/2,N_L,endpoint=True))
            self.z_position.extend(np.linspace(-d/2*np.tan(alpha/180*np.pi),0,N_L,endpoint=True))
            self.psi_angle.extend(np.ones(N_L)*(90+alpha))
        else:    #The AR given gives a flat bottom
            z_bottom=-d*AR
            x_bottom=(d/2*np.tan(alpha/180*np.pi)-np.abs(z_bottom))/np.tan(alpha/180*np.pi)    #this is the x index for the bottom edge
            L=(d/2-x_bottom)/np.cos(alpha/180*np.pi)
            N_L=int(L/step)
            self.x_position.extend(np.linspace(-d/2,-x_bottom,N_L,endpoint=False))
            self.z_position.extend(np.linspace(0,z_bottom,N_L,endpoint=False))
            self.psi_angle.extend(np.ones(N_L)*(90-alpha))
            N_bottom=int(x_bottom/step)
            self.x_position.extend(np.linspace(-x_bottom,0,N_bottom,endpoint=True))
            self.z_position.extend(np.ones(N_bottom)*z_bottom)
            self.psi_angle.extend(np.ones(N_bottom)*90)
            self.x_position.extend(np.linspace(0,x_bottom,N_bottom,endpoint=False))
            self.z_position.extend(np.ones(N_bottom)*z_bottom)
            self.psi_angle.extend(np.ones(N_bottom)*90)
            self.x_position.extend(np.linspace(x_bottom,d/2,N_L,endpoint=True))
            self.z_position.extend(np.linspace(z_bottom,0,N_L,endpoint=True))
            self.psi_angle.extend(np.ones(N_L)*(90+alpha))
        self.x_position.extend(np.linspace(d/2,d,N,endpoint=False))
        self.z_position.extend(np.zeros(N))
        self.psi_angle.extend(np.ones(N)*90)
        