import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import time as tm

start_time = tm.perf_counter()

#time initialization
dt = 0.15            # [s] time step for the simulation
TOTAL_TIME = 60            # [s] total time of the simulation
time_steps = int(np.floor(TOTAL_TIME/dt))

#fixed values
fs_bef = 0.6
tau = 0.24
thick_to_chord = 24.1

#importing data
total_data = np.loadtxt('C:\\COPENAGHEN PRIMO ANNO\\AEROELASTICITY\\turbine_data\\FFA-W3-241_ds.txt');
blade_data = np.loadtxt('C:\\COPENAGHEN PRIMO ANNO\\AEROELASTICITY\\turbine_data\\bladedat.txt');
radius = blade_data[:,0]
aoa = total_data[:,0]
lift_coefficient = total_data[:,1]
drag_coefficient = total_data[:,2]
separation_function = total_data[:,4]
linear_lift_coefficient = total_data[:,5]
stalled_lift_coefficient = total_data[:,6]

#initialize vectors
angle_attack = np.zeros(time_steps)   
clift_dyn_stall = np.zeros(time_steps)
fs_stat = np.zeros(time_steps)
for ii in range(time_steps):
    time = ii*dt
    angle_attack[ii] = 15+5*np.sin(time*np.pi/6)
    cdthick = np.interp(angle_attack[ii],aoa, drag_coefficient)
    fs_stat[ii] = np.interp(angle_attack[ii],aoa, separation_function)
    linear_clift = np.interp(angle_attack[ii],aoa, linear_lift_coefficient)
    stalled_clift = np.interp(angle_attack[ii],aoa, stalled_lift_coefficient)
    # cdthick = interpolate.interp1d(aoa, drag_coefficient,
    #                                    kind='linear')(angle_attack[ii])
    # fs_stat[ii]= interpolate.interp1d(aoa, separation_function,
    #                                    kind='linear')(angle_attack[ii])
    # linear_clift = interpolate.interp1d(aoa, linear_lift_coefficient,
    #                                    kind='linear')(angle_attack[ii])
    # stalled_clift = interpolate.interp1d(aoa, stalled_lift_coefficient,
    #                                    kind='linear')(angle_attack[ii])

    fs_now = fs_stat[ii]+(fs_bef-fs_stat[ii])*np.exp(-dt/tau)
    clift_dyn_stall[ii] = fs_now*linear_clift+(1-fs_now)*stalled_clift
    fs_bef = fs_now
fig = plt.figure(1)
plt.plot(angle_attack,clift_dyn_stall,linestyle='--',label=r'$Stalled\:C_{lift}$')
plt.plot(aoa[50:63],lift_coefficient[50:63],linestyle='-',label=r'$Airfoil\:lift$')
plt.legend(loc='best')

#calculating the execution time of the script
end_time = tm.perf_counter()
execution_time = end_time-start_time
print(f'The program required {execution_time:.3f} s')