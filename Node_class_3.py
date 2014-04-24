#-------------------------------------------------------------------------------
# Name:        Node_class_3
# Purpose:     to simulate reaction profile for two-molecule cvd in trenches, using given sticking coefficients profiles
#
# Author:      Wenjiao_Wang
#
# Created:     16/04/2014
# Copyright:   (c) Wenjiao_Wang 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import scipy as sp
from scipy.integrate import quad
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
import math
import cmath

out_trench_nodes=10
Cr=2    #critical flux, for flux lower than this value, the sticking coefficient is about a constant, 1/Cr, for flux higher than this value, the sticking coefficient is inverse-proportional to C.
f1r=1.0
f2r=0.2
max_node_spacing=1.0/50*2
min_node_spacing=1.0/50/2   #maintain the min length to be 1% of the max length

def make_complex_vector(real, imag):
    if len(real)!=len(imag):
        print 'Error! Sizes of vectors do not equal!'
        return 0
    else:
        result=np.zeros(len(real),dtype='complex')
        result.real=real
        result.imag=imag
        return result



class Nodes:

    def __init__(self, N=50, alpha=90.0, AR=10, Sc_surface=1):    #Assuming the trench openning width is unity. currently the trench class only supports side wall angle smaller than 90 degree. N stands for the number of nodes on one side of the trench wall.
        self.position = []    #the positions of the nodes in y-z plane, y is the real part and z is the imag part
        self.psi_angle = []

        self.AR=AR    #AR stands for the aspect ratio of the trench, is the depth/width. If this number is larger than d/2*tan(alpha), the trench is a 'V' shape, and has no flat bottom
        self.alpha=alpha    #alpha stands for the angle between the trench side wall and y axis. Currently, it can only be smaller than 90.
        self.N=N    #stands for the node density of the trench. N is the number of nodes on surface, outside the trench, for d/2 length.
        self.Sc_surface=Sc_surface  #this is the sticking coefficient on the surface outside the trench. The flux outside the trench is 1/Sc_surface
        self.position.extend(np.linspace(complex(-1.0,0),complex(-1.0/2,0),out_trench_nodes,endpoint=False))

        if np.tan(alpha/180.0*np.pi)<2*AR:    #the AR given is too large, and the trench is a 'V' shape, no flat part is included at the bottom

            Depth=1.0/2*np.tan(alpha/180.0*np.pi)   #Depth is the depth of the trench in the central point.
            self.position.extend(np.linspace(complex(-1.0/2,0),complex(0,-Depth),N+1,endpoint=True))
            self.position.extend(np.linspace(complex(1.0/2,0),complex(0,-Depth),N,endpoint=False)[::-1])

        else:    #The AR given gives a flat bottom

            y_bottom=(1.0/2*np.tan(alpha/180.0*np.pi)-AR)/np.tan(alpha/180.0*np.pi)    #this is the y index for the bottom edge
            self.position.extend(np.linspace(complex(-1.0/2,0),complex(-y_bottom,-AR),N,endpoint=False))
            self.position.extend(np.linspace(complex(-y_bottom,-AR),complex(0,-AR),N/2+1,endpoint=True))
            self.position.extend(np.linspace(complex(y_bottom,-AR),complex(0,-AR),N/2,endpoint=False)[::-1])
            self.position.extend(np.linspace(complex(1.0/2,0),complex(y_bottom,-AR),N,endpoint=False)[::-1])

        self.position.extend(np.linspace(complex(1,0),complex(1.0/2,0),out_trench_nodes,endpoint=False)[::-1])
          #here the position vector is a list, not an array
        self.renew_psi()
        self.center_positions=self.renew_center_positions()
        self.center_psi=self.renew_center_psi()

    def renew_psi(self):
        length=len(self.position)
        new_psi=np.zeros(length)
        for i in range(0,length/2,1):
            #print i
            new_psi[i]=cmath.phase(self.position[i+1]-self.position[i])/np.pi*180.0+90
            new_psi[length-1-i]=180.0-new_psi[i]
        new_psi[length/2]=90.0
        self.psi_angle=new_psi

    def renew_center_positions(self):
        length=len(self.position)
        center_positions=np.zeros(length-1,'complex')
        for i in range(0,length-1,1):
            center_positions[i]=(self.position[i]+self.position[i+1])/2
        return center_positions

    def renew_center_psi(self):
        length=len(self.position)
        center_psi=np.zeros(length-1)
        for i in range(0,length/2,1):
            center_psi[i]=self.psi_angle[i]
            center_psi[length-2-i]=180-center_psi[i]
        return center_psi

    def Node_inside_trench(self, node_index=1):    #'''determine whether one node is inside the trench or not'''
        if self.position[node_index].imag<self.trench_top():
            return True
        else:
            return False

    def center_inside_trench(self, node_index=1):
        if self.center_positions[node_index].imag<self.trench_top():
            return True
        else:
            return False

    def center_connecting_slope(self, node_index1, node_index2):    #to find the connecting slope between any two nodes, index1 is on the left, the slope is defined as the angle between vector from index1 to index2 and x axis
        slope=cmath.phase(self.position[node_index2]-self.center_positions[node_index1])    #from a center_position to a node position
        return slope/np.pi*180.0    #the angle between index1 to index2 and x axis

    def Node_connecting_slope(self, node_index1, node_index2):    #to find the connecting slope between any two nodes, index1 is on the left, the slope is defined as the angle between vector from index1 to index2 and x axis
        slope=cmath.phase(self.position[node_index2]-self.position[node_index1])    #from a center_position to a node position
        return slope/np.pi*180.0    #the angle between index1 to index2 and x axis

    def center_find_shadowing_point(self, node_index=1):    #to find the two points that defines the window of entrance flux to the current node, if it is inside the trench
        i=0
        while(not self.Node_inside_trench(i)):
            i+=1
        lb=i-1    #lb records the left bound of the trench


        i=len(self.position)-1

        while(not self.Node_inside_trench(i)):
            i-=1
        ub=i+1    #ub records the right bound of the trench

        if self.center_inside_trench(node_index):    #the center node under consideration is within the trench
            l_slope_min=180
            l_index=0
            r_slope_max=0
            r_index=0
            i=lb
            while(i<=node_index and self.position[i].imag>self.center_positions[node_index].imag):#the ith node should be on the left side of node_index and its depth should be higher than node_index
                slope_l=self.center_connecting_slope(node_index,i)
                if l_slope_min>slope_l:
                    l_slope_min=slope_l
                    l_index=i
                i+=1
            j=ub
            while(j>node_index and self.position[j].imag>self.center_positions[node_index].imag):
                slope_r=self.center_connecting_slope(node_index,j)
                if r_slope_max<slope_r:
                    r_slope_max=slope_r
                    r_index=j
                j-=1
            return [l_slope_min, r_slope_max,l_index,r_index]#the first two items are the slopes while the last two items are indices of the shadowing points
        else:
            print "This node is not inside the trench."
            return 0

    def Node_find_shadowing_point(self, node_index=1):    #to find the two points that defines the window of entrance flux to the current node, if it is inside the trench
        i=0
        while(not self.Node_inside_trench(i)):
            i+=1
        lb=i-1    #lb records the left bound of the trench


        i=len(self.position)-1

        while(not self.Node_inside_trench(i)):
            i-=1
        ub=i+1    #ub records the right bound of the trench

        if self.Node_inside_trench(node_index):    #the center node under consideration is within the trench
            l_slope_min=180
            l_index=0
            r_slope_max=0
            r_index=0
            i=lb
            while(i<=node_index and self.position[i].imag>self.position[node_index].imag):#the ith node should be on the left side of node_index and its depth should be higher than node_index
                slope_l=self.Node_connecting_slope(node_index,i)
                if l_slope_min>slope_l:
                    l_slope_min=slope_l
                    l_index=i
                i+=1
            j=ub
            while(j>node_index and self.position[j].imag>self.position[node_index].imag):
                slope_r=self.Node_connecting_slope(node_index,j)
                if r_slope_max<slope_r:
                    r_slope_max=slope_r
                    r_index=j
                j-=1
            return [l_slope_min, r_slope_max,l_index,r_index]#the first two items are the slopes while the last two items are indices of the shadowing points
        else:
            print "This node is not inside the trench."
            return 0

    def center_find_entrance_corners(self, node_index=1):    #this function gives the 4 points on the trench entrance plane that defines the entrance area that could give flux to the node_index point inside trench. K_number is the Knudsen number, and is defined as mean_free_path/trench_width.
        if self.center_inside_trench(node_index):
            [slope1, slope2, l_index, r_index]=self.center_find_shadowing_point(node_index)

            return [self.position[l_index].real,self.position[r_index].real]
        else:
            print "This node is not inside the trench."
            return 0

    def Node_find_entrance_corners(self, node_index=1):    #this function gives the 4 points on the trench entrance plane that defines the entrance area that could give flux to the node_index point inside trench. K_number is the Knudsen number, and is defined as mean_free_path/trench_width.
        if self.Node_inside_trench(node_index):
            [slope1, slope2, l_index, r_index]=self.Node_find_shadowing_point(node_index)

            return [self.position[l_index].real,self.position[r_index].real]
        else:
            print "This node is not inside the trench."
            return 0

    def trench_top(self):
        return self.position[0].imag   #returns the current trench top position in z axis, equals current thickness of the first node which is outside the trench

    def integrate_direct_flux(self, node_index=1):    #this function assumes infinity in x axis, and use the integrated function to calculate the direct flux, which is almost the same as integration, and is 100 times faster.
        e_1=1.0/(self.Sc_surface)
        if self.center_inside_trench(node_index):  # and yr!=np.inf and yl!=-np.inf:   #if yr or yl is inf, the node is outside the trench
            [yl,yr]=self.center_find_entrance_corners(node_index=node_index)	#first calculate for the center position
            #if yr!=np.inf and yl!=-np.inf:
            M=(yr-self.center_positions[node_index].real)/((self.trench_top()-self.center_positions[node_index].imag)**2*math.sqrt((yr-self.center_positions[node_index].real)**2+(self.trench_top()-self.center_positions[node_index].imag)**2))-(yl-self.center_positions[node_index].real)/((self.trench_top()-self.center_positions[node_index].imag)**2*math.sqrt((yl-self.center_positions[node_index].real)**2+(self.trench_top()-self.center_positions[node_index].imag)**2))
            N=1.0/math.sqrt((yl-self.center_positions[node_index].real)**2+(self.trench_top()-self.center_positions[node_index].imag)**2)-1.0/math.sqrt((yr-self.center_positions[node_index].real)**2+(self.trench_top()-self.center_positions[node_index].imag)**2)
            C_1=e_1*(self.trench_top()-self.center_positions[node_index].imag)*((-self.center_positions[node_index].real*np.cos(self.center_psi[node_index]/180.0*np.pi)-(self.center_positions[node_index].imag-self.trench_top())*np.sin(self.center_psi[node_index]/180.0*np.pi))*M/2.0+np.cos(self.center_psi[node_index]/180.0*np.pi)/2.0*(N+self.center_positions[node_index].real*M))
            #for the position at the beginning of the segment
            if self.Node_inside_trench(node_index):
                M_2=(yr-self.position[node_index].real)/((self.trench_top()-self.position[node_index].imag)**2*math.sqrt((yr-self.position[node_index].real)**2+(self.trench_top()-self.position[node_index].imag)**2))-(yl-self.position[node_index].real)/((self.trench_top()-self.position[node_index].imag)**2*math.sqrt((yl-self.position[node_index].real)**2+(self.trench_top()-self.position[node_index].imag)**2))
                N_2=1.0/math.sqrt((yl-self.position[node_index].real)**2+(self.trench_top()-self.position[node_index].imag)**2)-1.0/math.sqrt((yr-self.position[node_index].real)**2+(self.trench_top()-self.position[node_index].imag)**2)
                C_2=e_1*(self.trench_top()-self.position[node_index].imag)*((-self.position[node_index].real*np.cos(self.center_psi[node_index]/180.0*np.pi)-(self.position[node_index].imag-self.trench_top())*np.sin(self.center_psi[node_index]/180.0*np.pi))*M_2/2.0+np.cos(self.center_psi[node_index]/180.0*np.pi)/2.0*(N_2+self.position[node_index].real*M_2))
            else:
                C_2=e_1*(self.center_psi[node_index]+90.0)/180.0
            #for the position at the end of the segment
            M_3=(yr-self.position[node_index+1].real)/((self.trench_top()-self.position[node_index+1].imag)**2*math.sqrt((yr-self.position[node_index+1].real)**2+(self.trench_top()-self.position[node_index+1].imag)**2))-(yl-self.position[node_index+1].real)/((self.trench_top()-self.position[node_index+1].imag)**2*math.sqrt((yl-self.position[node_index+1].real)**2+(self.trench_top()-self.position[node_index+1].imag)**2))
            N_3=1.0/math.sqrt((yl-self.position[node_index+1].real)**2+(self.trench_top()-self.position[node_index+1].imag)**2)-1.0/math.sqrt((yr-self.position[node_index+1].real)**2+(self.trench_top()-self.position[node_index+1].imag)**2)
            C_3=e_1*(self.trench_top()-self.position[node_index+1].imag)*((-self.position[node_index+1].real*np.cos(self.center_psi[node_index]/180.0*np.pi)-(self.position[node_index+1].imag-self.trench_top())*np.sin(self.center_psi[node_index]/180.0*np.pi))*M_3/2.0+np.cos(self.center_psi[node_index]/180.0*np.pi)/2.0*(N_3+self.position[node_index+1].real*M_3))

            return 1.0/6*(C_2+C_3+4*C_1)
            #else:
                #return e_1
        else:
            return e_1

    def integrate_direct_flux_1(self, node_index,surface_flux_1):    #this function assumes infinity in x axis, and use the integrated function to calculate the direct flux, which is almost the same as integration, and is 100 times faster.
        e_1=surface_flux_1
        if self.center_inside_trench(node_index):  # and yr!=np.inf and yl!=-np.inf:   #if yr or yl is inf, the node is outside the trench
            [yl,yr]=self.center_find_entrance_corners(node_index=node_index)	#first calculate for the center position
            #if yr!=np.inf and yl!=-np.inf:
            M=(yr-self.center_positions[node_index].real)/((self.trench_top()-self.center_positions[node_index].imag)**2*math.sqrt((yr-self.center_positions[node_index].real)**2+(self.trench_top()-self.center_positions[node_index].imag)**2))-(yl-self.center_positions[node_index].real)/((self.trench_top()-self.center_positions[node_index].imag)**2*math.sqrt((yl-self.center_positions[node_index].real)**2+(self.trench_top()-self.center_positions[node_index].imag)**2))
            N=1.0/math.sqrt((yl-self.center_positions[node_index].real)**2+(self.trench_top()-self.center_positions[node_index].imag)**2)-1.0/math.sqrt((yr-self.center_positions[node_index].real)**2+(self.trench_top()-self.center_positions[node_index].imag)**2)
            C_1=e_1*(self.trench_top()-self.center_positions[node_index].imag)*((-self.center_positions[node_index].real*np.cos(self.center_psi[node_index]/180.0*np.pi)-(self.center_positions[node_index].imag-self.trench_top())*np.sin(self.center_psi[node_index]/180.0*np.pi))*M/2.0+np.cos(self.center_psi[node_index]/180.0*np.pi)/2.0*(N+self.center_positions[node_index].real*M))
            #for the position at the beginning of the segment
            if self.Node_inside_trench(node_index):
                M_2=(yr-self.position[node_index].real)/((self.trench_top()-self.position[node_index].imag)**2*math.sqrt((yr-self.position[node_index].real)**2+(self.trench_top()-self.position[node_index].imag)**2))-(yl-self.position[node_index].real)/((self.trench_top()-self.position[node_index].imag)**2*math.sqrt((yl-self.position[node_index].real)**2+(self.trench_top()-self.position[node_index].imag)**2))
                N_2=1.0/math.sqrt((yl-self.position[node_index].real)**2+(self.trench_top()-self.position[node_index].imag)**2)-1.0/math.sqrt((yr-self.position[node_index].real)**2+(self.trench_top()-self.position[node_index].imag)**2)
                C_2=e_1*(self.trench_top()-self.position[node_index].imag)*((-self.position[node_index].real*np.cos(self.center_psi[node_index]/180.0*np.pi)-(self.position[node_index].imag-self.trench_top())*np.sin(self.center_psi[node_index]/180.0*np.pi))*M_2/2.0+np.cos(self.center_psi[node_index]/180.0*np.pi)/2.0*(N_2+self.position[node_index].real*M_2))
            else:
                C_2=e_1*(self.center_psi[node_index]+90.0)/180.0
            #for the position at the end of the segment
            M_3=(yr-self.position[node_index+1].real)/((self.trench_top()-self.position[node_index+1].imag)**2*math.sqrt((yr-self.position[node_index+1].real)**2+(self.trench_top()-self.position[node_index+1].imag)**2))-(yl-self.position[node_index+1].real)/((self.trench_top()-self.position[node_index+1].imag)**2*math.sqrt((yl-self.position[node_index+1].real)**2+(self.trench_top()-self.position[node_index+1].imag)**2))
            N_3=1.0/math.sqrt((yl-self.position[node_index+1].real)**2+(self.trench_top()-self.position[node_index+1].imag)**2)-1.0/math.sqrt((yr-self.position[node_index+1].real)**2+(self.trench_top()-self.position[node_index+1].imag)**2)
            C_3=e_1*(self.trench_top()-self.position[node_index+1].imag)*((-self.position[node_index+1].real*np.cos(self.center_psi[node_index]/180.0*np.pi)-(self.position[node_index+1].imag-self.trench_top())*np.sin(self.center_psi[node_index]/180.0*np.pi))*M_3/2.0+np.cos(self.center_psi[node_index]/180.0*np.pi)/2.0*(N_3+self.position[node_index+1].real*M_3))

            return 1.0/6*(C_2+C_3+4*C_1)
            #else:
                #return e_1
        else:
            return e_1

    def direct_flux_distribution(self,surface_flux_1,surface_flux_2):   #calculates for the fluxes for two species
        index=0
        length=len(self.center_positions)
        direct_flux_1=np.ones(length)
        direct_flux_2=np.ones(length)
        while index<length/2: #calculate the left side, then the right side can be calculated by symmetry
            temp=self.integrate_direct_flux_1(node_index=index,surface_flux_1=surface_flux_1)
            direct_flux_1[index]=temp
            direct_flux_1[length-1-index]=direct_flux_1[index]
            temp1=self.integrate_direct_flux_1(node_index=index,surface_flux_1=surface_flux_2)
            direct_flux_2[index]=temp1
            direct_flux_2[length-1-index]=direct_flux_2[index]
            index+=1
        return [direct_flux_1,direct_flux_2]


    def Sticking_coefficient_1(self, flux1, flux2):
        return flux1*(1.0/f1r)/(1.0+(flux1/f1r)+(flux2/f2r))     #min(flux1,flux2)*4.0*(1.0/f1r)*(flux2/f2r)/(1.0+(flux1/f1r)+(flux2/f2r))**2
        #return 0.1

    def Sticking_coefficient_2(self, flux1, flux2):
        return 1    #min(flux1,flux2)*4.0*(1.0/f2r)*(flux1/f1r)/(1.0+(flux1/f1r)+(flux2/f2r))**2

    def Sticking_coefficient_vector_1(self, receiving_flux_vector_1,receiving_flux_vector_2):
        length=len(receiving_flux_vector_1)
        result=np.zeros(length)
        for i in range(0,length,1):
            result[i]=self.Sticking_coefficient_1(flux1=receiving_flux_vector_1[i],flux2=receiving_flux_vector_2[i])

        return result

    def Sticking_coefficient_vector_2(self, receiving_flux_vector_1,receiving_flux_vector_2):
        length=len(receiving_flux_vector_2)
        result=np.zeros(length)
        for i in range(0,length,1):
            result[i]=self.Sticking_coefficient_2(flux1=receiving_flux_vector_1[i],flux2=receiving_flux_vector_2[i])

        return result

    def visibility(self, index1, index2):   #if the slope of the vector [smaller(index1, index2)->any node between index1 and index2] is smaller than the slope of the vector [index1-> index2], then the visibility is 1.
        if index1==index2:
            return 0    #means not visible to each other
        else:
            left=np.min([index1,index2])
            right=np.max([index1,index2])
            if right==left+1 and self.center_psi[right]<=self.center_psi[left]:    #if the nodes are neighboring points
                return 0

            if self.center_positions[left].real==self.center_positions[right].real:
                for i in range(left+1,right,1):
                    if self.center_positions[i].real>=self.center_positions[left].real or self.position[i].real>=self.center_positions[left].real:
                        return 0
                return 1
            else:
                for i in range(left+1,right,1):
                    if self.center_positions[i].real==self.center_positions[left].real:
                        if self.center_positions[right].real<self.center_positions[left].real:
                            return 0

                    elif cmath.phase(self.center_positions[i]-self.center_positions[left])-cmath.phase(self.center_positions[right]-self.center_positions[left])>=0 or cmath.phase(self.position[i]-self.center_positions[left])-cmath.phase(self.center_positions[right]-self.center_positions[left])>=0 or cmath.phase(self.position[right]-self.center_positions[left])-cmath.phase(self.center_positions[right]-self.center_positions[left])>=0:   #(self.z_position[i]-self.z_position[left])/(self.y_position[i]-self.y_position[left])>=(self.z_position[right]-self.z_position[left])/(self.y_position[right]-self.y_position[left]):  #if the slope of left->i is not smaller than left-> right
                        return 0

                return 1



    def receiving_matrix_element(self, index1, index2):   #index1 is the receiving node.
        if self.visibility(index1,index2)==0 or self.center_psi[index1]==self.center_psi[index2]:   #if the visibility is 0, the receiving_matrix_element should be 0 as well.
            return 0
        else:
            angle_index1=self.center_psi[index1]/180.0*np.pi
            angle_index2=self.center_psi[index2]/180.0*np.pi
            n_index1=np.matrix([np.cos(angle_index1),np.sin(angle_index1)])
            n_index2=np.matrix([np.cos(angle_index2),np.sin(angle_index2)])
            P_index1_index2=np.matrix([[self.center_positions[index1].real-self.center_positions[index2].real,],[self.center_positions[index1].imag-self.center_positions[index2].imag,]])
            q_index2_index1=-1*(np.array(n_index1*P_index1_index2)[0][0])*(np.array(n_index2*P_index1_index2)[0][0])*0.5*(1.0/(math.sqrt((np.array(P_index1_index2)[0][0])**2+(np.array(P_index1_index2)[1][0])**2))**3)
            if q_index2_index1<0 and q_index2_index1>-1e-10:
                q_index2_index1=0
            elif q_index2_index1<=-1e-10:
                print 'Encountered negtive receiving_matrix_element'
                print index1, index2
                print q_index2_index1
                q_index2_index1=0
            #calculate for the begining position of the current segment
            if self.visibility(index1-1,index2):
                if self.Node_inside_trench(index1):
                    P_index1_index2_1=np.matrix([[self.position[index1].real-self.center_positions[index2].real,],[self.position[index1].imag-self.center_positions[index2].imag,]])
                    q_index2_index1_1=-1*(np.array(n_index1*P_index1_index2_1)[0][0])*(np.array(n_index2*P_index1_index2_1)[0][0])*0.5*(1.0/(math.sqrt((np.array(P_index1_index2_1)[0][0])**2+(np.array(P_index1_index2_1)[1][0])**2))**3)
                    if q_index2_index1_1<0 and q_index2_index1_1>-1e-10:
                        q_index2_index1_1=0
                    elif q_index2_index1_1<=-1e-10:
                        print 'Encountered negtive receiving_matrix_element for Node position'
                        print index1, index2
                        print q_index2_index1_1
                        q_index2_index1_1=0
                else:
                    P_index1_index2_1=np.matrix([[self.center_positions[index1].real-self.center_positions[index2].real,],[self.center_positions[index1].imag-self.center_positions[index2].imag,]])
                    q_index2_index1_1=-1*(np.array(n_index1*P_index1_index2)[0][0])*(np.array(n_index2*P_index1_index2)[0][0])*0.5*(1.0/(math.sqrt((np.array(P_index1_index2)[0][0])**2+(np.array(P_index1_index2)[1][0])**2))**3)
                    if q_index2_index1_1<0 and q_index2_index1>-1e-10:
                        q_index2_index1_1=0
                    elif q_index2_index1_1<=-1e-10:
                        print 'Encountered negtive receiving_matrix_element for Node position'
                        print index1, index2
                        print q_index2_index1_1
                        q_index2_index1_1=0
            else:
                q_index2_index1_1=0
            #calculate for the ending position of the current segment
            if self.visibility(index1+1,index2):
                P_index1_index2_2=np.matrix([[self.position[index1+1].real-self.center_positions[index2].real,],[self.position[index1+1].imag-self.center_positions[index2].imag,]])
                q_index2_index1_2=-1*(np.array(n_index1*P_index1_index2_2)[0][0])*(np.array(n_index2*P_index1_index2_2)[0][0])*0.5*(1.0/(math.sqrt((np.array(P_index1_index2_2)[0][0])**2+(np.array(P_index1_index2_2)[1][0])**2))**3)
                if q_index2_index1_2<0 and q_index2_index1_2>-1e-10:
                    q_index2_index1_2=0
                elif q_index2_index1_2<=-1e-10:
                    print 'Encountered negtive receiving_matrix_element for Node position'
                    print index1+1, index2
                    print q_index2_index1_2
                    q_index2_index1_2=0
            else:
                q_index2_index1_2=0
            return 1.0/6*(4*q_index2_index1+q_index2_index1_1+q_index2_index1_2)

    def receiving_matrix(self):
        length=len(self.center_positions)
        result=np.zeros([length,length])
        for i in range(0,length/2+1,1): #no matter the trench shape, the 0-length/2 is the left half of the trench, including the middle node
            for j in range(0,length,1):
                result[j][i]=self.receiving_matrix_element(index1=i,index2=j)

        for i in range(length/2+1,length,1):
            for j in range(0,length,1):
                result[j][i]=result[length-1-j][length-1-i]
        return result


    def node_length(self, index=1): #calculate the length of the segment of the node position at index, for the left half, the segment is on the right side of the node position, on the right half, the segment is on the left of the node position, for the central point, the segment length is zero.
        length=len(self.center_positions)
        if index<length/2:  #left half
            return math.sqrt((self.position[index+1].real-self.position[index].real)**2+(self.position[index+1].imag-self.position[index].imag)**2)
        elif index>=length/2:    #right half
            return math.sqrt((self.position[index-1].real-self.position[index].real)**2+(self.position[index-1].imag-self.position[index].imag)**2)
        else:   #central point
            return 0

    def node_lengths(self):
        length=len(self.center_positions)
        node_lengths=np.zeros(length)
        for i in range(0,length/2,1):
            node_lengths[i]=self.node_length(index=i)
            node_lengths[length-1-i]=node_lengths[i]
        return node_lengths


    def Emission_vector(self,receiving_flux_vector,sticking_coefficient_vector,segment_length_vector):
        length=len(receiving_flux_vector)
        #result=np.matrix(receiving_flux_vector)*(np.matrix(np.eye(length))-np.matrix(sticking_coefficient_vector)*np.matrix(np.eye(length)))*(np.matrix(segment_length_vector)*np.matrix(np.eye(length)))
        result=np.zeros(length)
        for i in range(0, length,1):
            result[i]=receiving_flux_vector[i]*(1-sticking_coefficient_vector[i])*segment_length_vector[i]
        return result   #return type is matrix, is a row vector

    def receiving_flux_vector(self, Emission_vector, receiving_matrix_1, direct_flux_distribution_1):
        return Emission_vector*receiving_matrix_1+direct_flux_distribution_1    #return type is matrix, a row vector

    def stable_receiving_flux_vector(self,surface_flux_1,surface_flux_2,sticking_coefficient_vector_1,sticking_coefficient_vector_2,initial_flux,number_of_cycles):   #here the sticking_coefficient_vector_1, and sticking_coefficient_vector_2 are sticking coefficients vectors for the two fluxes, respectively.
        length=len(self.center_positions)
        last_receiving_flux_vector=np.zeros(length)
        current_receiving_flux_vector=np.zeros(length)
        segment_length_vector=self.node_lengths()
        receiving_matrix=self.receiving_matrix()

        direct_flux_distribution=self.direct_flux_distribution(surface_flux_1,surface_flux_2)
        #direct_flux_distribution=initial_flux

        '''
        for i in range(0,length,1):     #found nan values for some position
            if direct_flux_distribution_1[i]==np.nan and direct_flux_distribution_1[i-1]!=np.nan and direct_flux_distribution_1[i+1]!=np.nan:
                direct_flux_distribution_1[i]=(direct_flux_distribution_1[i-1]+direct_flux_distribution_1[i+1])/2
        '''
        for i in range(0,number_of_cycles,1):     #set the max iteration cycle to 50
            #quality=0   #converging quality factor, 1 means needing further calculation, 0 means the last_flux and current_flux are close enough
            if i==0:
                #last_receiving_flux_vector_1=direct_flux_distribution[0]
                #last_receiving_flux_vector_2=direct_flux_distribution[1]
                last_receiving_flux_vector_1=initial_flux[0]
                last_receiving_flux_vector_2=initial_flux[1]
            else:
                #sticking_coefficient_vector_1=self.Sticking_coefficient_vector_1(last_receiving_flux_vector_1,last_receiving_flux_vector_2)
                #sticking_coefficient_vector_2=self.Sticking_coefficient_vector_2(last_receiving_flux_vector_1,last_receiving_flux_vector_2)
                emission_vector_1=np.zeros(length)
                emission_vector_2=np.zeros(length)
                for j in range(0,length,1):
                    emission_vector_1[j]=last_receiving_flux_vector_1[j]*(1-sticking_coefficient_vector_1[j])*segment_length_vector[j]
                    emission_vector_2[j]=last_receiving_flux_vector_2[j]*(1-sticking_coefficient_vector_2[j])*segment_length_vector[j]
                current_receiving_flux_vector_1=np.array(np.matrix(emission_vector_1)*np.matrix(receiving_matrix))[0]+direct_flux_distribution[0]
                current_receiving_flux_vector_2=np.array(np.matrix(emission_vector_2)*np.matrix(receiving_matrix))[0]+direct_flux_distribution[1]
                last_receiving_flux_vector_1=current_receiving_flux_vector_1
                last_receiving_flux_vector_2=current_receiving_flux_vector_2
                '''
                for j in range(0,length,1):
                    if np.abs((current_receiving_flux_vector_1[j]-last_receiving_flux_vector_1[j])/last_receiving_flux_vector_1[j])>0.01 or np.abs((current_receiving_flux_vector_2[j]-last_receiving_flux_vector_2[j])/last_receiving_flux_vector_2[j])>0.01:
                        quality=1
                        break
                    else:
                        pass
                if quality==1:
                    last_receiving_flux_vector_1=current_receiving_flux_vector_1
                    last_receiving_flux_vector_2=current_receiving_flux_vector_2
                else:
                    break
                '''
        '''
        if i>90:
            print "the receiving flux vector is not converging in ",i," cycles!"
            return [current_receiving_flux_vector_1,current_receiving_flux_vector_2]
        else:
            return [current_receiving_flux_vector_1,current_receiving_flux_vector_2]
        '''
        return [current_receiving_flux_vector_1,current_receiving_flux_vector_2]


    def stable_receiving_flux_vector_1(self,SC):   #with fixed sticking coefficients
        length=len(self.center_positions)
        last_receiving_flux_vector=np.zeros(length)
        current_receiving_flux_vector=np.zeros(length)
        segment_length_vector=self.node_lengths()
        receiving_matrix=self.receiving_matrix()

        direct_flux_distribution=self.direct_flux_distribution()

        '''
        for i in range(0,length,1):     #found nan values for some position
            if direct_flux_distribution_1[i]==np.nan and direct_flux_distribution_1[i-1]!=np.nan and direct_flux_distribution_1[i+1]!=np.nan:
                direct_flux_distribution_1[i]=(direct_flux_distribution_1[i-1]+direct_flux_distribution_1[i+1])/2
        '''

        for i in range(0,70,1):     #set the max iteration cycle to 50
            quality=0   #converging quality factor, 1 means needing further calculation, 0 means the last_flux and current_flux are close enough
            if i==0:
                last_receiving_flux_vector=direct_flux_distribution
            else:
                #sticking_coefficient_vector=self.Sticking_coefficient_vector(last_receiving_flux_vector)
                sticking_coefficient_vector=np.ones(length)*SC
                emission_vector=np.zeros(length)
                for j in range(0,length,1):
                    emission_vector[j]=last_receiving_flux_vector[j]*(1-sticking_coefficient_vector[j])*segment_length_vector[j]
                current_receiving_flux_vector=np.array(np.matrix(emission_vector)*np.matrix(receiving_matrix))[0]+direct_flux_distribution
                for j in range(0,length,1):
                    if ((current_receiving_flux_vector[j]-last_receiving_flux_vector[j])/last_receiving_flux_vector[j])>0.01:
                        quality=1
                        break
                    else:
                        pass
                if quality==1:
                    last_receiving_flux_vector=current_receiving_flux_vector
                else:
                    break
        if i>40:
            print "the receiving flux vector is not converging in 40 cycles!"
            return direct_flux_distribution
        else:
            return current_receiving_flux_vector

    def stable_receiving_flux_vector_2(self,SC):   #with fixed sticking coefficients, taking the inverse matrix method to calculate for infinity times re-emission
        length=len(self.center_positions)
        last_receiving_flux_vector=np.zeros(length)
        current_receiving_flux_vector=np.zeros(length)
        segment_length_vector=self.node_lengths()
        receiving_matrix=self.receiving_matrix()
        for i in range(0,length,1):
            receiving_matrix[i]=(1-SC)*segment_length_vector[i]*receiving_matrix[i]


        direct_flux_distribution=self.direct_flux_distribution()
        S=np.matrix(np.eye(length)-receiving_matrix).I
        current_receiving_flux_vector=np.array(np.matrix(direct_flux_distribution)*S)[0]
        return current_receiving_flux_vector


    def trench_update(self, receiving_flux_vector, sticking_coefficient_vector):
        length=len(self.center_positions)
        increment_vector=np.zeros(length)
        complex_increment_vector=np.zeros(length,'complex')
        new_position=np.zeros(length+1,'complex')
        new_y=np.zeros(length)
        new_z=np.zeros(length)
        merge_point=0   #defines whether the trench has merging point inside trench during coating
        i=0
        while i<length:   #calculate increment vector for the center positions
            increment_vector[i]=receiving_flux_vector[i]*sticking_coefficient_vector[i]
            angle=self.center_psi[i]/180.0*np.pi
            complex_increment_vector[i]=complex(increment_vector[i]*np.cos(angle),increment_vector[i]*np.sin(angle))
            i+=1

        i=0
        while i<(length+1)/2:
            if i>0 and i<length:
                new_position[i]=self.position[i]+(complex_increment_vector[i]+complex_increment_vector[i-1])/2
                #new_position[length-i].real=-new_position[i].real
                #new_position[length-i].imag=new_position[i].imag
                new_position[length-i]=new_position[i]-2*new_position[i].real
            elif i==0:
                new_position[i]=self.position[i]+(complex_increment_vector[i]+complex_increment_vector[length-1])/2
                #new_position[length-i].real=-new_position[i].real
                #new_position[length-i].imag=new_position[i].imag
                new_position[length-i]=new_position[i]-2*new_position[i].real
            if new_position[length-1-i].real<=new_position[i].real:   #if the symmetric points crosses over, that is the two points merged together
                merge_point=i
                break
            i+=1

        #new_position[(length+1)/2].real=0   #for the central point, assumes its growth rate is the same as its neighbor
        new_position[(length+1)/2]=complex(0,new_position[(length+1)/2-1].imag+np.tan(cmath.phase(new_position[(length+1)/2-1]-new_position[(length+1)/2-2]))*(0-new_position[(length+1)/2-1].real))  #(new_position[(length+1)/2-1].imag-new_position[(length+1)/2-2].imag)/(new_position[(length+1)/2-1].real-new_position[(length+1)/2-2].real)

        if merge_point==0:
            self.position=new_position
            self.renew_psi()
            self.center_positions=self.renew_center_positions()
            self.center_psi=self.renew_center_psi()
            self.deloop()
            self.optimize_node_length()
            #self.psi_angle=new_psi
            #self.node_optimization()
        else:
            length_1=(i)*2+1
            new_position_1=np.zeros(length_1,'complex')
            new_y_1=np.zeros(length_1)
            new_z_1=np.zeros(length_1)
            for j in range(0,i,1):
                new_position_1[j]=new_position[j]
                #new_position_1[length_1-1-j].real=-new_position_1[j].real
                #new_position_1[length_1-1-j].imag=new_position_1[j].imag
                new_position_1[length_1-1-j]=new_position_1[j]-2*new_position_1[j].real

            new_position_1[length_1/2]=complex(0,new_position_1[length_1/2-1].imag+(new_position_1[length_1/2-1].imag-new_position_1[length_1/2-2].imag)/(new_position_1[length_1/2-1].real-new_position_1[length_1/2-2].real)*(0-new_position_1[length_1/2-1].real))
            self.position=new_position_1
            self.renew_psi()
            self.center_positions=self.renew_center_positions()
            self.center_psi=self.renew_center_psi()
            self.deloop()
            self.optimize_node_length()
        return 0

    def deloop(self):
        length=len(self.position)
        lm=length/2 #initialize the left most node to be the central bottom
        while(self.position[lm-1].real<self.position[lm].real) and ((self.position[lm].real-self.position[lm-1].real)>(self.position[lm+1].real-self.position[lm].real)/2):#np.abs((cmath.phase(self.position[lm1-1]-self.position[lm1])-cmath.phase(self.position[lm1]-self.position[lm1+1]))/np.pi*180.0)<5:  #move the lm to the left most node of the bottom
            lm-=1
        #now the lm is the left most node of the bottom.
        for i in range(0,length/2,1):
            if self.Node_inside_trench(i):
                break
        bm=i   #now bm is the first node inside the trench
        while(self.position[bm+1].imag<self.position[bm].imag): #and ((self.position[bm].imag-self.position[bm+1].imag)>(self.position[bm-1].imag-self.position[bm].imag)/2):#np.abs((cmath.phase(self.position[bm1+1]-self.position[bm1])-cmath.phase(self.position[bm1]-self.position[bm1-1]))/np.pi*180.0)<5:
            bm+=1
        #Now the bm is the bottom most node of the left wall.

        if self.position[bm].real<self.position[lm].real and self.position[bm].imag>self.position[lm].imag:
            print "there is no loop for this profile!"
        elif bm==lm:    #if the bottom and left wall finds the same position at the left bottom corner, then that point should be removed.
            bm-=1
            lm+=1
            lm1=lm
            bm1=bm
            M=cmath.tan(cmath.phase(self.position[lm1]-self.position[lm1+1]))
            N=cmath.tan(cmath.phase(self.position[bm1]-self.position[bm1-1]))
            a=((self.position[bm1].imag-self.position[lm1].imag)+M*self.position[lm1].real-N*self.position[bm1].real)/(M-N)
            b=M*(a-self.position[lm1].real)+self.position[lm1].imag
            new_corner=complex(a,b) #this is the position of the new corner.
            if self.position[lm1].real<new_corner.real: #find the right node to start to cut off.
                lm1=lm1+1   #this is the node to keep
            if self.position[bm1].imag<new_corner.imag:
                bm1=bm1-1
            offset=0
            new_node_position=[]
            new_node_position.extend(self.position[0:bm1+1])
            new_node_position.extend([new_corner])
            new_node_position.extend(self.position[lm1:length-lm1])
            new_node_position.extend([complex(-1*new_corner.real,new_corner.imag)])
            new_node_position.extend(self.position[length-bm1-1:length-1])
            self.position=new_node_position
            self.renew_psi()
            self.center_positions=self.renew_center_positions()
            self.center_psi=self.renew_center_psi()
        else:
            lm1=lm
            bm1=bm
            while (self.node_distance(lm1,bm1)>self.node_distance(lm1+1,bm1)) : #requires the distance to bm is shorter and the direction change is not greater than 5 degree
                lm1+=1
            while(self.node_distance(lm1,bm1)>self.node_distance(lm1,bm1-1)):
                bm1-=1
            #now the lm1 and bm1 are the nearest nodes at the left bottom corner
            M=cmath.tan(cmath.phase(self.position[lm1]-self.position[lm1+1]))
            N=cmath.tan(cmath.phase(self.position[bm1]-self.position[bm1-1]))
            a=((self.position[bm1].imag-self.position[lm1].imag)+M*self.position[lm1].real-N*self.position[bm1].real)/(M-N)
            b=M*(a-self.position[lm1].real)+self.position[lm1].imag
            new_corner=complex(a,b) #this is the position of the new corner.
            if self.position[lm1].real<new_corner.real: #find the right node to start to cut off.
                lm1=lm1+1   #this is the node to keep
            if self.position[bm1].imag<new_corner.imag:
                bm1=bm1-1
            offset=0
            new_node_position=[]
            new_node_position.extend(self.position[0:bm1+1])
            new_node_position.extend([new_corner])
            new_node_position.extend(self.position[lm1:length-lm1])
            new_node_position.extend([complex(-1*new_corner.real,new_corner.imag)])
            new_node_position.extend(self.position[length-bm1-1:length-1])
            '''
            for i in range(bm1+1,lm1,1):
                del self.position[i-offset]

                del self.position[length-2*offset-i]
                offset+=1
            '''
            self.position=new_node_position
            self.renew_psi()
            self.center_positions=self.renew_center_positions()
            self.center_psi=self.renew_center_psi()

    def optimize_node_length(self):
        i=0
        new_position=[]
        while(i<len(self.position)/2):
            new_position.extend([self.position[i]])
            if self.node_distance(i,i+1)>max_node_spacing:
                if i>0:
                    angle=(cmath.phase(self.position[i]-self.position[i-1])+cmath.phase(self.position[i+1]-self.position[i]))/2
                else:
                    angle=(cmath.phase(self.position[i+1]-self.position[i]))/2
                if i>0:
                    length=self.node_distance(i,i+1)/2/np.cos(angle-cmath.phase(self.position[i]-self.position[i-1]))
                else:
                    length=self.node_distance(i,i+1)/2
                y_index=self.position[i].real+length*np.cos(angle)
                z_index=self.position[i].imag+length*np.sin(angle)
                new_position.extend([complex(y_index,z_index)])  #add an additional node between i and i+1
            if self.node_distance(i,i+1)<min_node_spacing:
                if self.node_distance(i,i+2)>max_node_spacing:
                    new_position.extend([(self.position[i]+self.position[i+2])/2])
                    i+=1    #omitting the i+1 node
                else:
                    i+=1
            i+=1
        new_position.extend(new_position[::-1]-2*np.array(new_position[::-1]).real)
        new_position.insert(len(new_position)/2,self.position[len(self.position)/2])  #add the bottom central point
        self.position=new_position
        self.renew_psi()
        self.center_positions=self.renew_center_positions()
        self.center_psi=self.renew_center_psi()
    '''
    def deloop_check(self):
        check_origin=complex(0,self.trench_top()+0.5)
        flag=0  #signifies whether the current trench profile has no loop or several dots on the same position. If so, the flag will be set to 1.
        temp=cmath.phase(check_origin-self.position[0])
        for i in range(1,len(self.position)/2,1):
            current_angle=cmath.phase(check_origin-self.position[i])
            if temp<current_angle:
                temp=current_angle
                continue
            else:
                flag=1
                break
        return flag

    def deloop_length_optimize(self):
        while self.deloop_check:
            self.deloop()
            self.optimize_node_length()
    '''

    def node_distance(self,index1,index2):
        distance=self.position[index1]-self.position[index2]
        return np.sqrt(distance.real**2+distance.imag**2)

    def node_optimization(self):    #optimize the node configuration, to more truly represent the trench shape
        changed=0   #notify whether the nodes of the trench have changed through the current optimization run
        new_position=[]     #records the position of new nodes
        new_number=[]       #records the number of new nodes at according position
        new_y=[]
        new_z=[]
        new_psi=[]
        original_length=len(self.z_position)
        for i in range(0,original_length/2,1):    #the original_length/2-1 is the node before the central point
            if i<original_length/2-1:
                delta_psi=self.psi_angle[i+1]-self.psi_angle[i]
                if np.abs(delta_psi)>10 and math.sqrt((self.y_position[i+1]-self.y_position[i])**2+(self.z_position[i+1]-self.z_position[i])**2)>10*min_node_spacing:    #if the i+1 and i node have psi difference bigger than 10 degree, and the distance between i and i+1 is larger than 100 times the minimum spacing. For the very local features, disregard it.
                    changed=1
                    n=int(np.abs(delta_psi)/10+1)    #number of new nodes to insert between i and i+1
                    d_psi=delta_psi/(n+1)     #step angle difference between neighboring new nodes
                    new_y_angle=np.zeros(n)     #to store the positions of new nodes
                    new_z_angle=np.zeros(n)
                    new_psi_angle=np.zeros(n)
                    min_node_spacing_1=2.0*min_node_spacing     #assume the length of the new segments is 2*min_node_spacing
                    L_1=np.matrix([0,0])    #calculation for the sum of the turning vectors
                    #L_sin=0
                    #L_cos=0
                    for j in range(1,int(n+1),1):
                        #L_sin+=np.abs(min_node_spacing_1*np.sin(d_psi*j/180.0*np.pi))
                        #L_cos+=np.abs(min_node_spacing_1*np.cos(d_psi*j/180.0*np.pi))
                        L_1=L_1+min_node_spacing_1*np.matrix([np.cos(d_psi*j/180.0*np.pi),np.sin(d_psi*j/180.0*np.pi)])

                    length_L_1=math.sqrt((np.array(L_1)[0][0])**2+(np.array(L_1)[0][1])**2) #length of the first new node to the end of the last new node, which is same as the new position for i+1 node
                    alpha_1=self.psi_angle[i+1]-self.psi_angle[i]
                    if alpha_1<0:
                        alpha_1=180.0+alpha_1   #if the turning angle is larger than 90 degree
                    elif alpha_1>0:
                        alpha_1=180.0-alpha_1
                    L=(length_L_1/2)/np.sin(np.abs(alpha_1)/2.0/180.0*np.pi)    #this is the length of the vector from the first new node to the original i+1 node, and the original i+1 node to the new i+1 node



                    original_i_1=np.matrix([self.y_position[i+1],self.z_position[i+1]]) #position for original i+1 node
                    angle_i_1=(self.psi_angle[i+1]-90)/180.0*np.pi  #angle of vector i+1 to i+2
                    unit_vector_i_1=np.matrix([np.cos(angle_i_1),np.sin(angle_i_1)])
                    new_i_1=original_i_1+L*unit_vector_i_1
                    self.y_position[i+1]=np.array(new_i_1)[0][0]
                    self.z_position[i+1]=np.array(new_i_1)[0][1]    #update the position of i+1 node

                    angle_i=(self.psi_angle[i]-90)/180.0*np.pi  #angle of vector i to i+1
                    unit_vector_i=np.matrix([np.cos(angle_i),np.sin(angle_i)])
                    n_new_1=original_i_1-L*unit_vector_i    #this is the position of the first new node
                    new_y_angle[0]=np.array(n_new_1)[0][0]
                    new_z_angle[0]=np.array(n_new_1)[0][1]
                    new_psi_angle[0]=self.psi_angle[i]+d_psi    #this is still the normal to the first new node
                    for k in range(2,int(n+1),1):
                        new_psi_angle[k-1]=new_psi_angle[k-2]+d_psi     #this is the normal to k-1 th node
                        temp_angle=(new_psi_angle[k-2]-90)/180.0*np.pi  #this is the direction of k-1 to k
                        temp_unit_vector=np.matrix([np.cos(temp_angle),np.sin(temp_angle)])
                        n_new_k_2=np.matrix([new_y_angle[k-2],new_z_angle[k-2]])    #this is the position of k-2
                        n_new_k_1=n_new_k_2+min_node_spacing_1*temp_unit_vector       #this is the position of k-1
                        new_y_angle[k-1]=np.array(n_new_k_1)[0][0]
                        new_z_angle[k-1]=np.array(n_new_k_1)[0][1]

                    new_position.extend([i])    #after i, insert n new nodes
                    new_number.extend([n])
                    new_y.extend(new_y_angle)
                    new_z.extend(new_z_angle)
                    new_psi.extend(new_psi_angle)

                elif math.sqrt((self.y_position[i+1]-self.y_position[i])**2+(self.z_position[i+1]-self.z_position[i])**2)>max_node_spacing: #if segment i is larger than the max_node_spacing
                    changed=1
                    new_position.extend([i])
                    new_number.extend([1])
                    new_y.extend([(self.y_position[i+1]+self.y_position[i])/2.0])
                    new_z.extend([(self.z_position[i+1]+self.z_position[i])/2.0])
                    new_psi.extend([self.psi_angle[i]])     #keeping the same normal as i node

                elif math.sqrt((self.y_position[i+1]-self.y_position[i])**2+(self.z_position[i+1]-self.z_position[i])**2)<min_node_spacing: #if segment i is smaller than the min_node_spacing
                    changed=1
                    new_position.extend([i])
                    new_number.extend([-1])     #means deleting the next node, the i+1 node
                    self.psi_angle[i]=cmath.phase(complex(self.y_position[i+1]-self.y_position[i-1],self.z_position[i+1]-self.z_position[i-1]))/np.pi*180.0+90  #np.arccos((self.y_position[i+2]-self.y_position[i])/math.sqrt((self.y_position[i+2]-self.y_position[i])**2+(self.z_position[i+2]-self.z_position[i])**2))/np.pi*180.0+90  #update the i node's psi value
            else:   #now i equals to original_length/2-1, and refers to the node before the central node, i+1
                psi_central=cmath.phase(complex(self.y_position[i+2]-self.y_position[i+1],self.z_position[i+2]-self.z_position[i+1]))/np.pi*180.0+90    #np.arctan((self.z_position[i+2]-self.z_position[i+1])/(self.y_position[i+2]-self.y_position[i+1]))/np.pi*180.0+90.0  #calculate for the normal of i+1 ->i+2 vector
                delta_psi=psi_central-self.psi_angle[i]
                if np.abs(delta_psi)>10 and math.sqrt((self.y_position[i+1]-self.y_position[i])**2+(self.z_position[i+1]-self.z_position[i])**2)>10*min_node_spacing:    #if the i+1 and i node have psi difference bigger than 10 degree
                    changed=1
                    n=int(np.abs(delta_psi)/10+1)    #number of new nodes to insert between i and i+1
                    if n%2==1:  #n needs to be an even number to keep the trench to be symmetric and there is a central node.
                        n+=1
                    d_psi=delta_psi/(n+1)     #step angle difference between neighboring new nodes
                    new_y_angle=np.zeros(n/2)     #to store the positions of new nodes
                    new_z_angle=np.zeros(n/2)
                    new_psi_angle=np.zeros(n/2)
                    min_node_spacing_1=2.0*min_node_spacing     #assume the length of the new segments is 2*min_node_spacing
                    L_1=np.matrix([0,0])    #calculation for the sum of the turning vectors
                    #L_sin=0
                    #L_cos=0
                    for j in range(1,int(n+1),1):
                        #L_sin+=np.abs(min_node_spacing_1*np.sin(d_psi*j/180.0*np.pi))
                        #L_cos+=np.abs(min_node_spacing_1*np.cos(d_psi*j/180.0*np.pi))
                        L_1=L_1+min_node_spacing_1*np.matrix([np.cos(d_psi*j/180.0*np.pi),np.sin(d_psi*j/180.0*np.pi)])

                    length_L_1=math.sqrt((np.array(L_1)[0][0])**2+(np.array(L_1)[0][1])**2) #length of the first new node to the end of the last new node, which is same as the new position for i+1 node
                    alpha_1=psi_central-self.psi_angle[i]
                    if alpha_1<0:
                        alpha_1=180.0+alpha_1   #if the turning angle is larger than 90 degree
                    elif alpha_1>0:
                        alpha_1=180.0-alpha_1
                    L=(length_L_1/2)/np.sin(np.abs(alpha_1)/2.0/180.0*np.pi)    #this is the length of the vector from the first new node to the original i+1 node, and the original i+1 node to the new i+1 node



                    original_i_1=np.matrix([self.y_position[i+1],self.z_position[i+1]]) #position for original i+1 node
                    #angle_i_1=(psi_central-90)/180.0*np.pi  #angle of vector i+1 to i+2
                    #unit_vector_i_1=np.matrix([np.cos(angle_i_1),np.sin(angle_i_1)])
                    #new_i_1=original_i_1+L*unit_vector_i_1
                    #self.y_position[i+1]=np.array(new_i_1)[0][0]
                    #self.z_position[i+1]=np.array(new_i_1)[0][1]    #update the position of i+1 node

                    angle_i=(self.psi_angle[i]-90)/180.0*np.pi  #angle of vector i to i+1
                    unit_vector_i=np.matrix([np.cos(angle_i),np.sin(angle_i)])
                    n_new_1=original_i_1-L*unit_vector_i    #this is the position of the first new node
                    new_y_angle[0]=np.array(n_new_1)[0][0]
                    new_z_angle[0]=np.array(n_new_1)[0][1]
                    new_psi_angle[0]=self.psi_angle[i]+d_psi    #this is still the normal to the first new node
                    for k in range(2,n/2+2,1):      #the new node with index n/2 is the new central point, the i+1
                        if k<n/2+1:                   #the new node on the left half
                            new_psi_angle[k-1]=new_psi_angle[k-2]+d_psi     #this is the normal to k-1 th node
                            temp_angle=(new_psi_angle[k-2]-90)/180.0*np.pi  #this is the direction of k-1 to k
                            temp_unit_vector=np.matrix([np.cos(temp_angle),np.sin(temp_angle)])
                            n_new_k_2=np.matrix([new_y_angle[k-2],new_z_angle[k-2]])    #this is the position of k-2
                            n_new_k_1=n_new_k_2+min_node_spacing_1*temp_unit_vector       #this is the position of k-1
                            if np.abs(np.array(n_new_k_1)[0][0])<1e-5:
                                new_y_angle[k-1]=0
                            else:
                                new_y_angle[k-1]=np.array(n_new_k_1)[0][0]
                            new_z_angle[k-1]=np.array(n_new_k_1)[0][1]
                        else:                       #the new position for the central node
                            #new_psi_angle[k-1]=new_psi_angle[k-2]+d_psi     #this is the normal to k-1 th node
                            temp_angle=(new_psi_angle[k-2]-90)/180.0*np.pi  #this is the direction of k-1 to k
                            temp_unit_vector=np.matrix([np.cos(temp_angle),np.sin(temp_angle)])
                            n_new_k_2=np.matrix([new_y_angle[k-2],new_z_angle[k-2]])    #this is the position of k-2
                            n_new_k_1=n_new_k_2+min_node_spacing_1*temp_unit_vector       #this is the position of k-1
                            if np.abs(np.array(n_new_k_1)[0][0])<1e-5:
                                self.y_position[i+1]=0
                            else:
                                self.y_position[i+1]=np.array(n_new_k_1)[0][0]
                            self.z_position[i+1]=np.array(n_new_k_1)[0][1]

                    new_position.extend([i])    #after i, insert n new nodes
                    new_number.extend([n/2])  #insert the left half, including the new central point
                    new_y.extend(new_y_angle[0:n/2])
                    new_z.extend(new_z_angle[0:n/2])
                    new_psi.extend(new_psi_angle[0:n/2])

                elif math.sqrt((self.y_position[i+1]-self.y_position[i])**2+(self.z_position[i+1]-self.z_position[i])**2)>max_node_spacing: #if segment i is larger than the max_node_spacing
                    changed=1
                    new_position.extend([i])
                    new_number.extend([1])
                    new_y.extend([(self.y_position[i+1]+self.y_position[i])/2.0])
                    new_z.extend([(self.z_position[i+1]+self.z_position[i])/2.0])
                    new_psi.extend([self.psi_angle[i]])     #keeping the same normal as i node

                elif math.sqrt((self.y_position[i+1]-self.y_position[i])**2+(self.z_position[i+1]-self.z_position[i])**2)<min_node_spacing: #if segment i is smaller than the min_node_spacing
                    changed=1
                    new_position.extend([i-1])  #if i+1 is the central point, delete the previous point
                    new_number.extend([-1])     #means deleting the next node, the i+1 node
                    self.psi_angle[i-1]=cmath.phase(complex(self.y_position[i+1]-self.y_position[i-1],self.z_position[i+1]-self.z_position[i-1]))/np.pi*180.0+90    #np.arccos((self.y_position[i+1]-self.y_position[i-1])/math.sqrt((self.y_position[i+1]-self.y_position[i-1])**2+(self.z_position[i+1]-self.z_position[i-1])**2))/np.pi*180.0+90  #update the i node's psi value



        if changed==1:  #the nodes of the trench need to be updated
            new_dimension=len(self.z_position)+2*sum(new_number)
            updated_y=np.zeros(new_dimension)
            updated_z=np.zeros(new_dimension)
            updated_psi=np.zeros(new_dimension)
            j=0 #index for the new arrays
            k=0 #index for the new_position
            i=0 #index for the original arrays
            x=0 #index for the inserting arrays
            while i<len(self.z_position)/2+1:   #index for the original arrays, including the central point
                updated_y[j]=self.y_position[i]
                updated_z[j]=self.z_position[i]
                updated_psi[j]=self.psi_angle[i]
                if k<len(new_position) and i==new_position[k]:  #position for inserting
                    m=new_number[k]     #the number of nodes for insertion
                    if m==-1:           #delete the next node on the original node list
                        i+=1            #omit the i+1 node
                    else:
                        mi=0
                        while mi<m:
                            j+=1
                            updated_y[j]=new_y[x]
                            updated_z[j]=new_z[x]
                            updated_psi[j]=new_psi[x]
                            x+=1
                            mi+=1

                    k+=1

                i+=1
                j+=1

            for i in range(0,new_dimension/2,1):            #this part is to update the psi angles for all nodes
                updated_y[new_dimension-1-i]=-updated_y[i]
                updated_z[new_dimension-1-i]=updated_z[i]
                '''
                if i>0:
                    updated_psi[i-1]=cmath.phase(complex(updated_y[i]-updated_y[i-1],updated_z[i]-updated_z[i-1]))/np.pi*180.0+90
                    updated_psi[new_dimension-i]=updated_psi[i-1]
            updated_psi[new_dimension/2-1]=cmath.phase(complex(updated_y[new_dimension/2]-updated_y[new_dimension/2-1],updated_z[new_dimension/2]-updated_z[new_dimension/2-1]))/np.pi*180.0+90
            updated_psi[new_dimension/2+1]=updated_psi[new_dimension/2-1]
            updated_psi[new_dimension/2]=90.0
            '''
            self.y_position=updated_y
            self.z_position=updated_z
            self.renew_psi()

        return changed
