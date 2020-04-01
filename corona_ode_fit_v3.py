# -*- coding: utf-8 -*-
#Abareru monster areba, tokokon buchinomeshi!
#Karayaku otakara areba, muriyari hitorijime!
#Daidanfuteki, denkousekka! Shouri wa watashi no tameni aru!

#Version ODE_dataFit_v01
#Includes a data validation procedure which terminates the script
#if one of the concentration vectors has a different dimension than
#the vector of times


#Import sys to interrupt script from running if a big problem is detected
import sys
#Import numpy for scientific computing and array and vector handling
import numpy as np
#Import matplotlib.pyplot to plot results
import matplotlib.pyplot as plt
#Import odeint to solve the differential equations
from scipy.integrate import odeint
#Import fmin utility
from scipy.optimize import fmin
#Import minimize utility. It allows specifying different optimization methods
from scipy.optimize import minimize


#0. The kinetic model

#==============================================================================

def eq(par, initial_cond, start_t, end_t, incr):
    #Here par, initial_cond, start_t, end_t and incr are local variables
    #To cast the scoring function, global variables must be passed as arguments
    #-time-grid
    t = gt
    #rates
    p0 = par[0]
    p1 = par[1]
    p2 = par[2]

    #differential equations system
    def myReaction(x, dt):
        dx0 = - p0 * x[0] * x[1] / n0
        dx1 = p0 * x[0] * x[1] / n0 - p1 * x[1] - p2 * x[1]
        dx2 = p1 * x[1] 
        dx3 = p2 * x[1]

        return [dx0, dx1, dx2, dx3]

    #integrate
    ds = odeint(myReaction, initial_cond, t)

    return (ds[:, 0], ds[:, 1], ds[:, 2], ds[:, 3], t)

#==============================================================================

#1.0 Get Data

#==============================================================================

#t_exp = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
#y0 = np.array([1, 1, 1, 1, 8, 11, 11, 16, 17, 25, 27, 38, 42, 51, 78, 103, 128, 193, 223, 246, 330, 442, 505, 603, 646, 709, 712, 787])
#y1 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 4, 5, 5, 12, 12, 16, 22, 27, 28, 39, 51, 51, 52, 72, 72, 80, 91, 228, 240])
#y2 = np.array([0.1, 0.1, 0.1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 6, 8, 12, 15, 19, 20, 26, 27])

#n0 = 1500



t_exp = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
y1 = np.array([1, 1, 1, 1, 8, 11, 11, 16, 17, 25, 27, 38, 42, 51, 78, 103, 128, 193, 223, 246, 330, 442, 505, 603, 646, 709, 712, 787])
n0 = 2*int(np.max(y1)/100)*100
print('n0 = ' + str(n0))
y2 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 4, 5, 5, 12, 12, 16, 22, 27, 28, 39, 51, 51, 52, 72, 72, 80, 91, 228, 240])
y3 = np.array([0.1, 0.1, 0.1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 6, 8, 12, 15, 19, 20, 26, 27])
y0 = n0 * (np.ones(len(t_exp))) - y1 - y2 - y3
deltas40m = - y1 - y2 - y3 
y040m = 40000000 * (np.ones(len(t_exp))) + deltas40m

print('y040m = ' + str(y040m))



#==============================================================================

#1.0-----------------------DATA VALIDATION-------------------------------------

#1.1 Verify that all the vectors have the same amount of elements
#THIS NEXT LINE MUST BE MODIFIED MANUALLY FOR EACH CASE

gConc_exp = np.array([y0, y1, y2, y3])
gn_points = len(t_exp)
gn_eq = len(gConc_exp)
gValidation = True

print('Number of points: ' + str(gn_points))
print('Number of concentrations tracked in time: ' + str(gn_eq))

#Fist we get nice and wet...
#Then we check that all the vectors have the same number of elements

for concentrations in gConc_exp:
    gValidation = (len(concentrations) == gn_points) * gValidation
    if (gValidation == False):
        print('Vectors do not have the same dimension.')
        print('Vector of time has: ' + str(gn_points) + ' elements.')
        print('Vector of concentration ' + str(concentrations))
        print('has ' + str(len(concentrations)) + ' elements instead.')
        print('TERMINATING PROCESS BECAUSE ONE OF THE VECTORS')
        print('HAS A DIFFERENT NUMBER OF ELEMENTS... YOU ARE WELCOME.')
        sys.exit()

        break

#-------------------------END DATA VALIDATION----------------------------------

#2. Set up Info for Model System

#==============================================================================

#model parameters
#These are inital guesses for the optimization problem
#Is there a way to properly initialize this automatically from the input data?

k0 = 10  #0.169308
k1 = 10  #0.011756
k2 = 10 #0.039407

rates = (k0, k1, k2)

#It's better to load initial conditions directly from the data :)'
gInitial_cond_exp = [y0[0], y1[0], y2[0], y3[0]]
gStart_t = t_exp[0]
gEnd_t = t_exp[-1]

#The amount of intervals depends on the amount of decimal places of the t
#vector from the experimental data

gIntervals = 1000
gt = np.linspace(gStart_t, gEnd_t, gIntervals)

#This instructions calculate which points of the model will be compared against
#points of the data with the same time coordinate
index0 = np.argmax(gt >= t_exp[0])
index1 = np.argmax(gt >= t_exp[1])
index2 = np.argmax(gt >= t_exp[2])
index3 = np.argmax(gt >= t_exp[3])
index4 = np.argmax(gt >= t_exp[4])
index5 = np.argmax(gt >= t_exp[5])
index6 = np.argmax(gt >= t_exp[6])
index7 = np.argmax(gt >= t_exp[7])
index8 = np.argmax(gt >= t_exp[8])
index9 = np.argmax(gt >= t_exp[9])
index10 = np.argmax(gt >= t_exp[10])
index11 = np.argmax(gt >= t_exp[11])
index12 = np.argmax(gt >= t_exp[12])
index13 = np.argmax(gt >= t_exp[13])
index14 = np.argmax(gt >= t_exp[14])
index15 = np.argmax(gt >= t_exp[15])
index16 = np.argmax(gt >= t_exp[16])
index17 = np.argmax(gt >= t_exp[17])
index18 = np.argmax(gt >= t_exp[18])
index19 = np.argmax(gt >= t_exp[19])
index20 = np.argmax(gt >= t_exp[20])
index21 = np.argmax(gt >= t_exp[21])
index22 = np.argmax(gt >= t_exp[22])
index23 = np.argmax(gt >= t_exp[23])
index24 = np.argmax(gt >= t_exp[24])
index25 = np.argmax(gt >= t_exp[25])
index26 = np.argmax(gt >= t_exp[26])
index27 = np.argmax(gt >= t_exp[27])

gIndex = (int(index0), int(index1), int(index2),
     int(index3), int(index4), int(index5), int(index6), int(index7), int(index8), int(index9), int(index10),
     int(index11), int(index12), int(index13), int(index14), int(index15), int(index15), int(index17),
     int(index18), int(index19), int(index20), int(index21), int(index22), int(index23), int(index24),
     int(index25), int(index26), int(index27) )
gIndex = np.array(gIndex)

#==============================================================================

#3 Score Fit of System

#==============================================================================

def score(parameters):
    #a.Get solution to system
    m = eq(parameters, gInitial_cond_exp, gStart_t, gEnd_t, gIntervals)
    dif0 = m[0][gIndex] - y0
    dif1 = m[1][gIndex] - y1
    dif2 = m[2][gIndex] - y2
    dif3 = m[3][gIndex] - y3
    

    error0 = np.linalg.norm(dif0)
    error1 = np.linalg.norm(dif1)
    error2 = np.linalg.norm(dif2)
    error3 = np.linalg.norm(dif3)
    

    return (error0 + error1 + error2 + error3)

print(str(gStart_t))
print(str(gEnd_t))
print(str(gIndex))

fiteo = score(rates)

print('fiteo: ' + str(fiteo))

#Minimizing using Nelder-Mead simplex algorithm
#From documentation of fmin about NM:
#"In practice it can have poor performance in high-dimensional problems
# and is not robust to minimizing complicated functions."

#answ = fmin(score, (rates), full_output=0, maxiter=100000)

#For SQP optimization, constraints are defined here
#cons = ({'type': 'eq', 'fun': lambda x: circunferencia(x)})
#Bounds for setting all the parameters as positive values
#No need to define inequality constraints this way

gBnds = ((0, None), (0, None), (0, None))
gRes = minimize(score, rates, method='SLSQP',
    bounds=gBnds, options={'disp': False})


print('res_minimizeSQP: ' + str(gRes))
result = gRes.x

print('solution: ' + str(result))

solution = eq(gRes.x, gInitial_cond_exp, gStart_t, gEnd_t, gIntervals)

deltas40mcontinua = - solution[1] - solution[2] - solution[3]
y040mcontinua = 40000000 * (np.ones(len(solution[0]))) + deltas40mcontinua

#print('y040mcontinua = ' + str(y040mcontinua))

#plt.plot(solution[4], y040mcontinua)


fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

ax1.set_xlabel('X data')
ax1.set_ylabel('Y1 data', color='g')
ax2.set_ylabel('Y2 data', color='b')

ax2.plot(t_exp, y040m, marker='o', linestyle='None', color='b')
ax2.plot(solution[4], y040mcontinua, linestyle='-', color='b')
ax1.plot(t_exp, y1, marker='o', linestyle='None', color='r')
ax1.plot(solution[4], solution[1], linestyle='-', color='r')
ax1.plot(t_exp, y2, marker='o', linestyle='None', color='g')
ax1.plot(solution[4], solution[2], linestyle='-', color='g')
ax1.plot(t_exp, y3, marker='o', linestyle='None', color='c')
ax1.plot(solution[4], solution[3], linestyle='-', color='c')


#plt.plot(t_exp, y0, marker='o', linestyle='None', color='b')
#plt.plot(solution[4], solution[0], linestyle='-', color='b')
#plt.plot(t_exp, y1, marker='o', linestyle='None', color='r')
#plt.plot(solution[4], solution[1], linestyle='-', color='r')
#plt.plot(t_exp, y2, marker='o', linestyle='None', color='g')
#plt.plot(solution[4], solution[2], linestyle='-', color='g')
#plt.plot(t_exp, y3, marker='o', linestyle='None', color='c')
#plt.plot(solution[4], solution[3], linestyle='-', color='c')

plt.xlabel('Días desde el inicio de casos')
plt.ylabel('Número de casos reportados')

plt.show()