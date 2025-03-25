'''
This file is an attempt to do Aeroelasticity's Assignment 1 with the implementation
of classes.
'''

#%% IMPORT PACKAGES
import os
from pathlib import Path
# Imposta la cartella di lavoro alla cartella contenente questo file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
import time as tm
from transformation_matrixes import TransformationMatrixes
from position_definition import PositionDefinition
from undisturbed_wind_speed import UndisturbedWindSpeed
from induced_wind import InducedWind
from loads_calculation import LoadsCalculation
from saving_data import SavingData
from adding_turbulence import AddingTurbulence

#%% directory commands
current_dir = Path(__file__).resolve().parent
current_ass_dir = current_dir.parent
exercises_dir = current_ass_dir.parent
aeroelasticity_dir = exercises_dir.parent 
#%% plot commands
start_time = tm.perf_counter()

#size
mpl.rcParams['figure.figsize'] = (16,8)

#font size of label, title, and legend
mpl.rcParams['font.size'] = 25
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.labelsize'] = 25
mpl.rcParams['axes.titlesize'] = 30
mpl.rcParams['legend.fontsize'] = 25

#Lines and markers
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 9
mpl.rcParams['scatter.marker'] = "+"
plt_marker = "d"

#Latex font
plt.rcParams['font.family'] = 'serif'  # Simula il font di LaTeX
plt.rcParams['mathtext.fontset'] = 'cm'  # Usa Computer Modern per la matematica

#Export
mpl.rcParams['savefig.bbox'] = "tight"
    
#%% MAIN
if __name__ == "__main__":
    #%% Initialize values

    B = 3                        # [-] number of blades
    RATED_POWER = 10*1e6          # [MW] rated power
    V_IN = 4                      # [m/s] cut in speed 
    V_OUT = 25                    # [m/s] cut-out speed
    RHO = 1.225                   # [kg/m^3] density 
    omega = 0.72                 # [rad/s] rotational speed 
    theta_cone = np.deg2rad(0)    # [rad] cone angle
    theta_yaw = np.deg2rad(0)     # [rad] yaw angle
    theta_pitch = np.deg2rad(0)   # [rad] shaft's pitch angle
    H =119                        # [m] hub height
    L_s = 7.1                     # [m] shaft length 
    R = 89.15                     # [m] turbine's radius
    d_angle = np.deg2rad(6)       # [rad] angle step for the simulation
    dt = d_angle/omega            # [s] time step for the simulation
    TOTAL_TIME = 50              # [s] total time of the simulation
    time_steps = int(np.floor(TOTAL_TIME/dt))
    shear_exponent = 0       # [-] velocity profile's shear exponent
    ws_hub_height = 8             # [m/s] hub height wind speed
    heights_tow = np.array([0,11.5,23,34.5,46,57.5,69,80.5,92,103.5,115.63])
    a_x = np.array([8.3,8.0215,7.7431,7.4646,7.1861,6.9076,6.6291,6.3507,6.0722,\
                5.7937,5.5])/2      # [m] tower's radius
    TIP_PITCH = np.zeros(time_steps)  # [rad] tip pitch
    Kp = 1.5                      # [rad/(rad/s)] proportional pitch gain
    Ki = 0.64                     # [rad/rad] integral pitch gain
    KK = 14                       # [deg] gain reduction
    w_ref = 1.01                  # [rad/s] reference rotational speed
    # If I want to determine the axial induction factor locally for each
    # radial position I put True, otherwise to use a mean value from the entire
    # turbine I switch to False
    local_a_calculation = True    # local calculation of axial induction factor
    # For the third question, the pitch angle changes from 0 to 2 degrees for 
    # a certain period to see the reaction's delay in the power.
    third_point = False #want to have pitch changing
    stall_model = True
    adding_turbulence = False #want to add turbulence to the wind speed
    tower_shadow = True
    if third_point:
        TIP_PITCH[(np.arange(time_steps)*dt>=100) & (np.arange(time_steps)*dt <= 150)] = np.deg2rad(2)
    
    #%% tower height linear regression
    X = heights_tow.reshape(-1, 1)  # Independent variable
    y = a_x  # Dependent variable
    height_model = LinearRegression()
    height_model.fit(X, y)
    gain = height_model.coef_[0]  # slope
    offset = height_model.intercept_  # intercept
    #%% Opening file and saving the contents
    DATA = SavingData(number_of_airfoils = 6,path=aeroelasticity_dir)
    DATA.opening_files()
    aoa,lift_coefficient,drag_coefficient,separation_function,linear_lift_coefficient,\
           stalled_lift_coefficient,radius = DATA.storing_data()
    thickness_to_chord = np.array((100, 60, 48, 36, 30.1, 24.1))
    
    #%% turbulence creation
    # the turbulence files are created with the Mann model 
    turb_file_1 = r'..\..\turbulence_generator\sim1.bin'
    turb_file_2 = r'..\..\turbulence_generator\sim2.bin'
    turb_file_3 = r'..\..\turbulence_generator\sim3.bin'
    ADDING_TURBULENCE = AddingTurbulence(dt, time_steps, ws_hub_height)
    
    # saving the streamwise, normal and vertical turbulence
    w_turb_plane,x_turb,y_turb = ADDING_TURBULENCE.calculating_turbulence_field(turb_file_1)
    v_turb_plane,x_turb,y_turb = ADDING_TURBULENCE.calculating_turbulence_field(turb_file_2)
    u_turb_plane,x_turb,y_turb = ADDING_TURBULENCE.calculating_turbulence_field(turb_file_3)
    #%% vector inizialization
    position_blades = np.zeros((time_steps,B,3)) #position of the considered point
                                                 #in space
    V0_system1 = np.zeros((time_steps,B,3)) #velocity in ground system
    V0_system4 = np.zeros((time_steps,B,3)) #velocity in blade system
    W_induced_quasi_steady = np.zeros((time_steps,B,radius.shape[0],3)) #quasi-steady
                                                                        #induced velocity
    W_induced_intermediate = np.zeros((time_steps,B,radius.shape[0],3)) #intermediate induced
                                                                        #velocity used for
                                                                        #dynamic wake calculation
    U_turb = np.zeros((time_steps,B,radius.shape[0],3)) # turbulence velocity                                                                                                                              
    W_induced = np.zeros((time_steps,B,radius.shape[0],3)) #induced wind velocity
    f_s = np.zeros((time_steps,B,radius.shape[0])) #separation function for dynamic wake
    azimuthal_angle_blade1 = np.zeros((time_steps,1)) #angle position of blade 1
    time_array = np.linspace(0, TOTAL_TIME, time_steps) #time array
    rot_steps = time_array*omega/(2*np.pi)
    theta_blade = np.zeros((B,1)) #angle position of the three blades
    final_tangential_force = np.zeros((time_steps,radius.shape[0],B)) #p_t
    final_normal_force = np.zeros((time_steps,radius.shape[0],B)) #p_n
    torque = np.zeros((time_steps)) #torque over time
    power = np.zeros((time_steps)) #power over time
    thrust = np.zeros((time_steps)) #thrust over time
    thrust_first_blade = np.zeros((time_steps))
    thrust_second_blade = np.zeros((time_steps))
    thrust_third_blade = np.zeros((time_steps))
    pz_blade1_turb = np.zeros((time_steps))
    #initialization of axial and tangential induction factors
    a_values = np.zeros((time_steps,1)) #axial induction factor
    a_mean = 0 #initialization if a global induction factor is chosen
    w_induced_65m = np.zeros((time_steps)) # induced wind at 65.75 m along the blade
    wind_speed = np.zeros(time_steps)
    pitch_angle = np.zeros(time_steps) # current pitch angle
    pitch_angle_set = np.zeros(time_steps) # setpoint pitch angle
    pitch_angle_i = np.zeros(time_steps) # integral pitch angle
    pitch_angle_p = np.zeros(time_steps) # proportional pitch angle
    angular_velocity = np.zeros(time_steps)
    
    #%% class initialization
    TRANSFORMATION_MATR = TransformationMatrixes(theta_cone,theta_yaw,theta_pitch)
    POSITION = PositionDefinition(TRANSFORMATION_MATR,omega,H,L_s)
    UNDISTURBED_WIND = UndisturbedWindSpeed(TRANSFORMATION_MATR,H,shear_exponent)
    INDUCED_WIND = InducedWind(B,RHO,TIP_PITCH,omega, R)
    LOADS_CALCULATION = LoadsCalculation(aoa, lift_coefficient,drag_coefficient,\
                             separation_function,linear_lift_coefficient,\
                             stalled_lift_coefficient,DATA.number_of_airfoils,\
                             thickness_to_chord,B,RATED_POWER,V_IN,V_OUT,RHO,\
                             TIP_PITCH,omega,theta_cone,theta_yaw,theta_pitch,
                             H,L_s,R,shear_exponent)
    
    #%% time loop
    for ii in range(0,time_steps):
            #initialize p_n and p_t for each time step
            tangential_force = np.zeros((radius.shape[0],B))
            normal_force = np.zeros((radius.shape[0],B))
            
            #time update
            time = ii*dt
            
            # # omega update
            # if ii!=0:
            #     omega = angular_velocity[ii]
                
            #saving the position of the blades at current time
            theta_blade[0],theta_blade[1],theta_blade[2] = POSITION.position_blade(time) 
            azimuthal_angle_blade1[ii] = theta_blade[0]
            
            #loop on the number of blades
            for jj in range(0,B):
                    theta_blade_considered = float(theta_blade[jj].item())
                    
                    #loop on every element of the blade
                    for kk in range(0,radius.shape[0]-1):
                        
                        #calculates the velocity in system 1 and 4
                        position_blades[ii,jj,:],a_1,a_2,a_12,a_23,a_34,a_14,a_21,\
                                  a_41 = POSITION.final_calculation_position\
                                         (theta_blade_considered, radius[kk])
                        if adding_turbulence: # I add the turbulence to the V0 in system 1
                            # interpolating the position of blade's element with
                            # the turbulence plane for each component
                            # U_turb[ii,jj,kk,0] = ADDING_TURBULENCE.interpolating_turbulence\
                            #                     (ii, position_blades[ii,jj,0],\
                            #                     position_blades[ii,jj,1],u_turb_plane)
                            # U_turb[ii,jj,kk,1] = ADDING_TURBULENCE.interpolating_turbulence\
                            #                     (ii, position_blades[ii,jj,0],\
                            #                     position_blades[ii,jj,1],v_turb_plane)
                            U_turb[ii,jj,kk,2] = ADDING_TURBULENCE.interpolating_turbulence\
                                                (ii, position_blades[ii,jj,0],\
                                                position_blades[ii,jj,1],w_turb_plane)
                            
                            V0_system1[ii,jj,:] = UNDISTURBED_WIND.velocity_system1(
                                                        position_blades[ii,jj,:],\
                                                        ws_hub_height,gain,offset,tower_shadow)\
                                                        +U_turb[ii,jj,kk,:]
                        else:
                            V0_system1[ii,jj,:] = UNDISTURBED_WIND.velocity_system1(
                                                        position_blades[ii,jj,:],\
                                                        ws_hub_height,gain,offset,tower_shadow)
                        V0_system4[ii,jj,:] = UNDISTURBED_WIND.velocity_system4(\
                                                        V0_system1[ii,jj,:])
                        
                        #data of the considered element
                        twist = np.deg2rad(DATA.blade_data[kk,1])
                        chord = DATA.blade_data[kk,2]
                        thick_to_chord = DATA.blade_data[kk,3]
                        
                        #calculation relative velocity. In the first iteration the
                        #induced wind is set to 0, as also the separation function.
                        #Hence, there will be a small unsteady period where it'll
                        #converge to the right value.
                        if ii == 0:
                            V_rel = LOADS_CALCULATION.relative_velocity(V0_system4\
                                                  [ii,jj,:], [0, 0, 0], radius[kk])
                            W_z = 0
                            lift,phi,normal_force[kk,jj],tangential_force[kk,jj],f_s[ii,jj,kk] = \
                                LOADS_CALCULATION.calculation_loads(V_rel,twist, chord,\
                                                     thick_to_chord,ii,0,dt,time,stall_model)
                        else:
                            V_rel = LOADS_CALCULATION.relative_velocity(V0_system4\
                                                  [ii,jj,:], W_induced[ii-1,jj,kk,:],\
                                                  radius[kk])
                            W_z = W_induced[ii-1,jj,kk,2]
                            lift,phi,normal_force[kk,jj],tangential_force[kk,jj],f_s[ii,jj,kk] = \
                                LOADS_CALCULATION.calculation_loads(V_rel,twist, chord,\
                                                     thick_to_chord,ii,f_s[ii-1,jj,kk],dt,time,\
                                                     stall_model)
                                    
                        #induced wind calculation with dynamic wake
                        F = INDUCED_WIND.tip_loss_correction(radius[kk], phi)
                        f_g,a = INDUCED_WIND.Glauert_correction(ii,W_z,V0_system4[ii,jj,:],\
                                                a_mean,local_a_calculation)
                        induced_denominator = INDUCED_WIND.denominator_induced_wind\
                                                (ii, V0_system4[ii,jj,:],f_g, W_z)
                        W_induced_quasi_steady[ii,jj,kk,:] = INDUCED_WIND.induced_wind_quasi_steady\
                                                (ii,lift,phi, radius[kk], F, induced_denominator)
                        

                        if time < 5:
                            tau_1,tau_2 = (1e-3,1e-3)
                        else:
                            tau_1,tau_2 = INDUCED_WIND.time_constants_induced_wind(\
                                                    a,ws_hub_height,radius[kk])
                        W_induced[ii,jj,kk,:],W_induced_intermediate[ii,jj,kk,:] = INDUCED_WIND.dynamic_wake\
                                                    (W_induced_quasi_steady[ii,jj,kk,:],\
                                                    W_induced_quasi_steady[ii-1,jj,kk,:],\
                                                    W_induced_intermediate[ii-1,jj,kk,:],\
                                                    W_induced[ii-1,jj,kk,:],
                                                    tau_1,tau_2,dt)   
            # mean value of a on the blades (calculated as the value at 0.7*R).
            if ii!=0:
                a_mean = (-W_induced[ii-1,:,8,2]/ws_hub_height).mean()
                a_values[ii] = a_mean
            
            # pitch controller
            
            #save the p_n, p_t, torque, power, trust array for the current time step
            pz_blade1_turb[ii] = normal_force[8,0]
            w_induced_65m[ii] = np.sqrt(W_induced[ii,0,8,0]**2+W_induced[ii,0,8,1]**2\
                                        +W_induced[ii,0,8,2]**2)
            final_tangential_force[ii,:,:] = tangential_force 
            final_normal_force[ii,:,:] = normal_force 
            torque[ii] = np.trapezoid(radius*tangential_force[:,0],radius)+\
                         np.trapezoid(radius*tangential_force[:,1],radius)+\
                         np.trapezoid(radius*tangential_force[:,2],radius)
            power[ii] = torque[ii]*omega;
            thrust_first_blade[ii] =  np.trapezoid(normal_force[:,0],radius)
            thrust_second_blade[ii] =  np.trapezoid(normal_force[:,1],radius)
            thrust_third_blade[ii] =  np.trapezoid(normal_force[:,2],radius)
            thrust[ii] = thrust_first_blade[ii] +\
                         thrust_second_blade[ii]+\
                         thrust_third_blade[ii]

#%% check spectrum
f_pz,PSD_pz = ADDING_TURBULENCE.calculating_psd(pz_blade1_turb,pwelch=False)
f_thrust,PSD_thrust = ADDING_TURBULENCE.calculating_psd(power,pwelch=True)
f_wind, PSD_wind = ADDING_TURBULENCE.calculating_psd(V0_system1[:,0,2],pwelch=True)
#%% results BEM
BEM_pz = np.loadtxt(r'../results/question1/BEM_pz.txt')
BEM_py = np.loadtxt(r'../results/question1/BEM_py.txt')
#%% FIGURES
colors = ['#377eb8','#e41a1c']

# Power and Thrust over time
fig,ax = plt.subplots(1,1)
ax.plot(rot_steps,power*1e-6,linestyle='--',label='Power',color = 'k')
ax2 = ax.twinx()
ax2.plot(rot_steps,thrust*1e-6,linestyle='-',label='Thrust',color = 'k')
handles, labels = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
combined_handles = handles + handles2
combined_labels = labels + labels2
leg = ax.legend(combined_handles, combined_labels, loc="upper left", frameon=False)
leg.get_frame().set_alpha(1)
ax.set_xlabel(r'complete rotations')
#plt.ylabel(r'$T\:[MN]$')
ax.set_ylabel(r'$P\:[MW]$')
ax2.set_ylabel(r'$T\:[MN]$')
ax.set_xlim([rot_steps[10], rot_steps[-1]])
ax.set_ylim([0.5,5])
ax.minorticks_on()
ax.grid()
ax.tick_params(direction='in',right=True,top =True)
ax.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
ax.tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
ax.tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)
# fig.savefig(r'../results/question1/power_thrust_si_tower_no_shear.pdf')

# Thrust over time with turbulence
# fig,ax = plt.subplots(1,1)
# ax.plot(time_array,thrust*1e-6,linestyle='-',label='Thrust',color = 'k')
# ax.set_xlabel(r't [s]')
# ax.set_ylabel(r'$T\:[MN]$')
# ax.set_xlim([time_array[10], time_array[-1]])
# #ax.set_ylim([0.5,5])
# ax.minorticks_on()
# ax.grid()
# ax.tick_params(direction='in',right=True,top =True)
# ax.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
# ax.tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
# ax.tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)
# fig.savefig(r'../results/question4/thrust_si_turbulence.pdf')

# Tangential and normal loads at the last iteration
# fig = plt.figure()
# plt.plot(radius,final_tangential_force[-1,:,0],linestyle='--',label=r'$P_y\:unsteady\:\:BEM$',color = 'k')
# plt.plot(radius,BEM_py,linestyle=':',marker='o',mfc='w',label=r'$P_y\:steady\:BEM$',color = 'k')
# plt.plot(radius,final_normal_force[-1,:,0],linestyle='-.',label=r'$P_z\:unsteady\:\:BEM$',color = 'k')
# plt.plot(radius,BEM_pz,linestyle=':',marker='d',mfc='w',label=r'$P_z\:steady\:BEM$',color = 'k')
# leg = plt.legend(loc="upper left", frameon=True)
# leg.get_frame().set_alpha(1)
# plt.xlabel(r'$r\: [m]$')
# plt.ylabel(r'$p\:[N]$')
# plt.xlim([radius[0], radius[-1]])
# plt.grid()
# plt.minorticks_on()
# plt.tick_params(direction='in',right=True,top =True)
# plt.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
# plt.tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
# plt.tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)
# plt.savefig(r'../results/question1/pz_py_last_iteration.pdf')

#PSD of normal loads at 65.5 m for blade number 1
#x= f_pz*2*np.pi/omega
# x= f_pz
# fig = plt.figure()
# plt.plot(x,PSD_pz,linestyle='-',color = 'k',label=r'$p_z\:PSD\:at\:65.75 m$')
# plt.yscale('log')
# plt.xscale('log')
#plt.xlabel(r'$f*2*\pi/\omega$')
# plt.xlabel(r'$f$')
# plt.ylabel(r'$PSD$')
# #plt.xlim([f_pz[0], f_pz[-1]])
# #plt.xlim([0,3.5])
# #plt.legend(loc="upper right",frameon= False)
# plt.grid()
# # plt.ylim([2,4])
# plt.minorticks_on()
# plt.tick_params(direction='in',right=True,top =True)
# plt.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
# plt.tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
# plt.tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)
# plt.savefig(r'../results/question4/PSD_pz_65m_si_turbulence_si_shear_frequency_visibles.pdf')

# # PSD of total thrust 
# fig = plt.figure()
# plt.plot(f_thrust*2*np.pi/omega,PSD_thrust,linestyle='-',color = 'k',label=r'Thrust PSD')
# plt.xlabel(r'$f*2*\pi/\omega$')
# plt.ylabel(r'$PSD$')
# plt.xlim([f_thrust[0], f_thrust[-1]])
# #plt.xlim([0,3.5])
# #plt.legend(loc="upper center",frameon= False)
# # plt.ylim([2,4])
# plt.grid()
# plt.minorticks_on()
# plt.tick_params(direction='in',right=True,top =True)
# plt.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
# plt.tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
# plt.tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)
# plt.savefig(r'../results/question4/PSD_thrust_si_turbulence_no_shear.pdf')


# normal loads at 65.5 m for blade number 1 over time
# fig = plt.figure()
# plt.plot(time_array,pz_blade1_turb,linestyle='-',color = 'k',label=r'$p_z\:at\:65.75\:m$')
# plt.xlabel(r't [s]')
# plt.ylabel(r'$p_z\:[N]$')
# #plt.legend(loc="upper left",frameon= False)
# plt.xlim([time_array[10], time_array[-1]])
# # plt.ylim([2,4])
# plt.grid()
# plt.minorticks_on()
# plt.tick_params(direction='in',right=True,top =True)
# plt.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
# plt.tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
# plt.tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)
# plt.savefig(r'../results/question4/pz_65m_si_turbulence.pdf')

# turbulent signal at 65.5 m for blade number 1 over time
# fig,axs = plt.subplots(3,1,sharex=True)
# fig.subplots_adjust(hspace=0.3)
# axs[0].plot(time_array,U_turb[:,0,8,0],linestyle='-',color = 'k')
# axs[1].plot(time_array,U_turb[:,0,8,1],linestyle='-',color = 'k')
# axs[2].plot(time_array,U_turb[:,0,8,2],linestyle='-',color = 'k')
# axs[0].set_title('Vertical fluctuations')
# axs[1].set_title('Transverse fluctuations')
# axs[2].set_title('Streamwise fluctuations')
# axs[2].set_xlabel(r'$t\: [s]$')
# axs[0].set_ylabel('V [m/s]')
# axs[1].set_ylabel('V [m/s]')
# axs[2].set_ylabel('V [m/s]')
# axs[2].set_xlim([time_array[0], time_array[-1]])
# axs[0].set_ylim([-1.3,1.3])
# axs[1].set_ylim([-1.3,1.3])
# axs[2].set_ylim([-1.3,1.3])
# axs[0].minorticks_on()
# axs[0].tick_params(direction='in',right=True,top =True)
# axs[0].tick_params(labelbottom=False,labeltop=False,labelleft=True,labelright=False)
# axs[0].tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
# axs[0].tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)
# axs[1].minorticks_on()
# axs[1].tick_params(direction='in',right=True,top =True)
# axs[1].tick_params(labelbottom=False,labeltop=False,labelleft=True,labelright=False)
# axs[1].tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
# axs[1].tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)
# axs[2].minorticks_on()
# axs[2].tick_params(direction='in',right=True,top =True)
# axs[2].tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
# axs[2].tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
# axs[2].tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)
# #fig.suptitle('Turbulent Velocity for B=1 and r=65.75 m')
# plt.savefig(r'../results/question4/turbulent_fluctuations.pdf')

#induced wind on blade 1 at 65.75 m
# fig = plt.figure()
# plt.plot(rot_steps,w_induced_65m,linestyle='--',color = 'k',label='induced wind at 65.75 m')
# plt.legend(loc="upper left",frameon= False)
# plt.xlabel(r'complete rotations')
# plt.ylabel(r'$W\:[m/s]$')
# plt.xlim([rot_steps[5], rot_steps[-1]])
# plt.ylim([2,4])
# plt.minorticks_on()
# plt.tick_params(direction='in',right=True,top =True)
# plt.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
# plt.tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
# plt.tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)
# plt.savefig(r'../results/question3/induced_wind_changed_pitch.pdf')
#%% calculating the execution time of the script

end_time = tm.perf_counter()
execution_time = end_time-start_time
print(f'The program required {execution_time:.1f} s')