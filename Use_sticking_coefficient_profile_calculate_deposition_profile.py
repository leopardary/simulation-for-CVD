#-------------------------------------------------------------------------------
# Name:        Use_sticking_coefficient_profile_calculate_deposition_profile
# Purpose:
#
# Author:      Wenjiao_Wang
#
# Created:     21/04/2014
# Copyright:   (c) Wenjiao_Wang 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import arange, array, exp
from Node_class_3 import Nodes

def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return array(map(pointwise, array(xs)))

    return ufunclike

Sc_H2O=np.array([0.0541,0.0592,0.0650,0.0717,0.0794,0.0883,0.0984,0.1100,0.1229,0.1366,0.1497,0.1591,0.1596,0.1478,0.1278,0.1065,0.0874,0.0713,0.0582,0.0475,0.0388,0.0317,0.0260,0.0217,0.0184,0.0161,0.0147,0.0143])*1e-3
Sc_Mg=np.array([0.0686,0.0787,0.0911,0.1065,0.1260,0.1509,0.1828,0.2243,0.2782,0.3476,0.4339,0.5316,0.6219,0.6786,0.6968,0.6930,0.6813,0.6683,0.6562,0.6458,0.6369,0.6295,0.6236,0.6190,0.6155,0.6131,0.6116,0.6112])*1e-3
AR=np.array([0,1.6933,3.3866,5.0800,6.7733,8.4666,10.1599,11.8532,13.5466,15.2399,16.9332,18.6265,20.3198,22.0132,23.7065,25.3998,27.0931,28.7864,30.4798,32.1731,33.8664,35.5597,37.2530,38.9464,40.6397,42.3330,44.0263,45.7196])

Sc_H2O_1=interp1d(AR,Sc_H2O)
Sc_Mg_1=interp1d(AR,Sc_Mg)
Sc_H2O_2=extrap1d(Sc_H2O_1)
Sc_Mg_2=extrap1d(Sc_Mg_1)

AR_new=np.linspace(0,60,61)

plt.plot(AR,Sc_H2O,'ro',AR_new,Sc_H2O_2(AR_new),'b-')
plt.plot(AR,Sc_Mg,'ro',AR_new,Sc_Mg_2(AR_new),'b-')
plt.show()



Flux=1
Sc_surface=1.0/Flux
C_sticking=0.1  #defines a constant sticking coefficient
trench=Nodes(N=100,alpha=90.0,AR=60,Sc_surface=Sc_surface)
[a,b]=trench.direct_flux_distribution(3.0098*0.1*1e3,7.2362*0.1*1e3)
plt.plot(a,'bo')
plt.plot(b,'ro')
plt.show()

trench_position=np.array(trench.position)
AR_depth=trench.trench_top()-trench_position.imag
sticking_H2O=Sc_H2O_2(AR_depth)
sticking_Mg=Sc_Mg_2(AR_depth)

[F_H2O,F_Mg]=trench.stable_receiving_flux_vector(3.0098*0.1*1e4/np.sqrt(18),7.2362*0.1*1e4/np.sqrt(167.58),sticking_H2O,sticking_Mg)
D_H2O=np.zeros(len(F_H2O))
D_Mg=np.zeros(len(F_Mg))
for i in range(0,len(F_H2O),1):
    D_H2O[i]=F_H2O[i]*sticking_H2O[i]
    D_Mg[i]=F_Mg[i]*sticking_Mg[i]

plt.plot(D_H2O,'ro',D_Mg,'bo')
plt.show()
