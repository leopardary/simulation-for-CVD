import numpy as np
import scipy as sp
from scipy.integrate import quad
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
import math

class Nodes:
    def __init__(self,d=100.0, N=100, alpha=90.0, AR=10, K_number=10, Sc_surface=1):    #currently the trench class only supports side wall angle smaller than 90 degree.
        self.y_position = []    #x index is zero for the nodes on this cross-section of trench
        self.z_position = []
        self.psi_angle = []
        self.d=d    #d stands for the trench opening width
        self.AR=AR    #AR stands for the aspect ratio of the trench, is the depth/width. If this number is larger than d/2*tan(alpha), the trench is a 'V' shape, and has no flat bottom
        self.alpha=alpha    #alpha stands for the angle between the trench side wall and x axis. Currently, it can only be smaller than 90.
        self.N=N    #stands for the node density of the trench. N is the number of nodes on surface, outside the trench, for d/2 length.
        self.K_number=K_number  #this is the Knudsen number for the trench under certain pressure condition
        self.Sc_surface=Sc_surface  #this is the sticking coefficient on the surface outside the trench.

        self.y_position.extend(np.linspace(-d,-d/2,N,endpoint=True)) #building the initial nodes for the bare trench
        self.z_position.extend(np.zeros(N))
        self.psi_angle.extend(np.ones(N)*90)
        step = d/2/(N-1)
        if np.tan(alpha/180*np.pi)<2*AR:    #the AR given is too large, and the trench is a 'V' shape, no flat part is included at the bottom
            L = d/2/np.cos(alpha/180*np.pi)    #L is the length of the trench wall
            N_L=int(L/step)    #N_L is the number of points on the trench wall, while keeping relatively the same point density
            self.y_position.extend(np.linspace(-d/2,0,N_L,endpoint=False))
            self.z_position.extend(np.linspace(0,-d/2*np.tan(alpha/180*np.pi),N_L,endpoint=False))
            self.psi_angle.extend(np.ones(N_L)*(90-alpha))
            self.y_position.extend(np.linspace(0,d/2,N_L,endpoint=True))
            self.z_position.extend(np.linspace(-d/2*np.tan(alpha/180*np.pi),0,N_L,endpoint=True))
            self.psi_angle.extend(np.ones(N_L)*(90+alpha))
        else:    #The AR given gives a flat bottom
            z_bottom=-d*AR
            y_bottom=(d/2*np.tan(alpha/180*np.pi)-np.abs(z_bottom))/np.tan(alpha/180*np.pi)    #this is the y index for the bottom edge
            L=(d/2-y_bottom)/np.cos(alpha/180*np.pi)
            N_L=int(L/step)
            self.y_position.extend(np.linspace(-d/2,-y_bottom,N_L,endpoint=False))
            self.z_position.extend(np.linspace(0,z_bottom,N_L,endpoint=False))
            self.psi_angle.extend(np.ones(N_L)*(90-alpha))
            N_bottom=int(y_bottom/step)
            self.y_position.extend(np.linspace(-y_bottom,0,N_bottom,endpoint=True))
            self.z_position.extend(np.ones(N_bottom)*z_bottom)
            self.psi_angle.extend(np.ones(N_bottom)*90)
            self.y_position.extend(np.linspace(0,y_bottom,N_bottom,endpoint=False))
            self.z_position.extend(np.ones(N_bottom)*z_bottom)
            self.psi_angle.extend(np.ones(N_bottom)*90)
            self.y_position.extend(np.linspace(y_bottom,d/2,N_L,endpoint=True))
            self.z_position.extend(np.linspace(z_bottom,0,N_L,endpoint=True))
            self.psi_angle.extend(np.ones(N_L)*(90+alpha))
        self.y_position.extend(np.linspace(d/2,d,N,endpoint=False))
        self.z_position.extend(np.zeros(N))
        self.psi_angle.extend(np.ones(N)*90)

    def inside_trench(self, node_index=1):    #'''determine whether one node is inside the trench or not'''
        if self.z_position[node_index]<self.z_position[0]:
            return True
        else:
            return False

    def connecting_slope(self, node_index1, node_index2):    #to find the connecting slope between any two nodes, index1 is on the left, the slope is defined as the angle between vector from index1 to index2 and x axis
        delta_z=(self.z_position[node_index2]-self.z_position[node_index1])
        delta_y=(self.y_position[node_index2]-self.y_position[node_index1])
        slope=np.arccos(delta_y/math.sqrt(delta_y**2+delta_z**2))
        return slope/np.pi*180    #the angle between index1 to index2 and x axis


    def find_shadowing_point(self, node_index=1):    #to find the two points that defines the window of entrance flux to the current node, if it is inside the trench
        i=0
        while(not self.inside_trench(i)):
            i+=1
        lb=i-1    #lb records the left bound of the trench

        i=len(self.y_position)-1

        while(not self.inside_trench(i)):
            i-=1
        ub=i+1    #ub records the right bound of the trench

        if (node_index>lb and node_index<ub):    #the node under consideration is within the trench
            l_slope_max=0
            l_index=0
            r_slope_max=0
            r_index=0
            i=lb
            while(i<node_index and self.z_position[i]>self.z_position[node_index]):#the ith node should be on the left side of node_index and its depth should be higher than node_index
                slope_l=self.connecting_slope(i,node_index)
                if l_slope_max+0.01<slope_l:
                    l_slope_max=slope_l
                    l_index=i
                i+=1
            j=ub
            while(j>node_index and self.z_position[j]>self.z_position[node_index]):
                slope_r=self.connecting_slope(node_index,j)
                if r_slope_max+0.01<slope_r:
                    r_slope_max=slope_r
                    r_index=j
                j-=1
            return [l_slope_max, r_slope_max]
        else:
            print "This node is not inside the trench."
            return 0

    def find_entrance_corners(self, node_index=1):    #this function gives the 4 points on the trench entrance plane that defines the entrance area that could give flux to the node_index point inside trench. K_number is the Knudsen number, and is defined as mean_free_path/trench_width.
        if self.inside_trench(node_index):
            r=self.d*self.K_number    #this is the mean_free_path, which stands for the longest possible path that a molecule can travel from outside the trench to a node inside the trench.
            [slope1, slope2]=self.find_shadowing_point(node_index)
            yl=self.y_position[node_index]-(self.z_position[1]-self.z_position[node_index])/np.tan(slope1/180*np.pi)    #the x index for the crossing on the left side wall
            yr=self.y_position[node_index]+(self.z_position[1]-self.z_position[node_index])/np.tan(slope2/180*np.pi)    # the x index for the crossing on the right side wall
            xl=math.sqrt(r**2-(yl-self.y_position[node_index])**2-(self.z_position[1]-self.z_position[node_index])**2)
            xr=math.sqrt(r**2-(yr-self.y_position[node_index])**2-(self.z_position[1]-self.z_position[node_index])**2)
            return [xl,yl,xr,yr]
        else:
            print "This node is not inside the trench."
            return 0

    def trench_top(self):
        return self.z_position[1]    #returns the current trench top position in z axis, equals current thickness of the first node which is outside the trench

    def integrate_direct_flux(self, node_index=1):
        if self.inside_trench(node_index):
            [xl,yl,xr,yr]=self.find_entrance_corners(node_index=node_index)
            #inte_result_1=quad(lambda y: ((2*(xl-xr)*(y-yr)/(yl-yr)+2*xr)/(2*((y-self.y_position[node_index])**2+(self.trench_top()-self.z_position[node_index])**2)*((y-self.y_position[node_index])**2+(self.trench_top()-self.z_position[node_index])**2+((xl-xr)*(y-yr)/(yl-yr)+xr)**2))+1/(2*math.sqrt((y-self.y_position[node_index])**2+(self.trench_top()-self.z_position[node_index])**2)*((y-self.y_position[node_index])**2+(self.trench_top()-self.z_position[node_index])**2))*np.arctan((2*(xl-xr)*(y-yr)/(yl-yr)+2*xr)/math.sqrt((y-self.y_position[node_index])**2+(self.trench_top()-self.z_position[node_index])**2))*(self.trench_top()-self.z_position[node_index])*((y-self.y_position[node_index])*np.cos(self.psi_angle[node_index])+(self.trench_top()-self.z_position[node_index])*np.sin(self.psi_angle[node_index]))),yl,yr)    #this is what I derived for integration with x part, but the result seems not in agreement with direct integration
            inte_result_2=dblquad(lambda y, x: (np.abs(self.trench_top()-self.z_position[node_index])*(np.abs((y-self.y_position[node_index])*np.cos(self.psi_angle[node_index]/180*np.pi))+np.abs((self.trench_top()-self.z_position[node_index])*np.sin(self.psi_angle[node_index]/180*np.pi)))/((x**2+(y-self.y_position[node_index])**2+(self.trench_top()-self.z_position[node_index])**2)**2)), yl, yr, lambda y: -(xl-xr)*(y-yl)/(yl-yr)-xl, lambda y: (xl-xr)*(y-yl)/(yl-yr)+xl)    #
            #inte_result_2=dblquad(lambda y, x: ((trench.trench_top()-trench.z_position[node_index])*((y-trench.y_position[node_index])*np.cos(trench.psi_angle[node_index]/180*np.pi)+(trench.trench_top()-trench.z_position[node_index])*np.sin(trench.psi_angle[node_index]/180*np.pi))/(x**2+(y-trench.y_position[node_index])**2+(trench.trench_top()-trench.z_position[node_index])**2)**2), yl, yr, lambda y: -(xl-xr)*(y-yr)/(yl-yr)-xr, lambda y: (xl-xr)*(y-yr)/(yl-yr)+xr) #test this statement
            G=1/(np.pi*self.Sc_surface) #G is the prefactor for calculating direct flux for different nodes, assuming that the direct flux to nodes outside the trench is 1.
            return [inte_result_2[0]*G, inte_result_2[1]*G]    #, inte_result_1
        else:
            return [1,0]    #if the node under consideration is outside the trench, the direct flux is assumed to be 1


