'''
This file is an attempt to do Aeroelasticity's Assignment 2 with the implementation
of classes.
'''

#%% IMPORT PACKAGES
import os
from pathlib import Path
# Imposta la cartella di lavoro alla cartella contenente questo file
os.chdir(Path(__file__).resolve().parent)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml
from sklearn.linear_model import LinearRegression
import time as tm
from transformation_matrixes import TransformationMatrixes
from position_definition import PositionDefinition
from undisturbed_wind_speed import UndisturbedWindSpeed
from induced_wind import InducedWind
from loads_calculation import LoadsCalculation
from saving_data import SavingData
from adding_turbulence import AddingTurbulence
from pitch_controller import PitchController

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

#%% DTU 10 MW report values
rpm_report = np.array([*[6.0]*4, 6.426, 7.229, 8.032 , 8.836, *[9.6]*14])
C_p_report = np.array([.286, .418, .464, .478, .476, .476, .476, .476, .402,
                       .317, .253, .207, .170, .142, .119, .102, .087, .075,
                       .065, .057, .05, .044])
theta_p_report = np.array([2.751, 1.966, 0.896, 0, 0, 0, 0, 0, 4.4502, 7.266,
                           9.292, 10.958, 12.499, 13.896, 15.2, 16.432, 17.618,
                           18.758, 19.860, 20.927, 21.963, 22.975])
P_mech_report = np.array([280.2, 799.1, 1532.7, 2506.1 , 3730.7, 5311.8,
                          7286.5, 9698.3, 10639.1, 10648.5, 10639.3, 10683.7,
                          10642.0, 10640.0, 10639.9, 10652.8, 10646.2, 10644.0,
                          10641.2, 10639.5, 10643.6, 10635.7])
#%% MAIN
if __name__ == "__main__":
    #%% Initialize values
    # Turbine parameters
    B = 3                         # [-] number of blades
    RATED_POWER = 10.64e6         # [MW] rated power
    V_IN = 4                      # [m/s] cut in speed 
    V_OUT = 25                    # [m/s] cut-out speed
    H =119                        # [m] hub height
    L_s = 7.1                     # [m] shaft length 
    R = 89.15                     # [m] turbine's radius
    Kp = 1.5                      # [rad/(rad/s)] proportional pitch gain
    Ki = 0.64                     # [rad/rad] integral pitch gain
    KK = 14                       # [deg] gain reduction
    w_ref = 9.62*np.pi/30;                  # [rad/s] reference rotational speed
    max_change_pitch = 10         # [deg/s] maximum pitch change
    pitch_max = 45                # [deg] maximum pitch of the blades
    pitch_min = 0                 # [deg] minimum pitch of the blades
    I_rot = 1.6e8                 # [kg*m^2] rotor inertia
    rated_omega = 9.6*np.pi/30;           # [rad/s] rotational speed at rated power
    rated_velocity = 11.4260;     # [m/s] wind speed at rated power
    heights_tow = np.array([0,11.5,23,34.5,46,57.5,69,80.5,92,103.5,115.63]) # tower heights
    a_x = np.array([8.3,8.0215,7.7431,7.4646,7.1861,6.9076,6.6291,6.3507,6.0722,\
                5.7937,5.5])/2    # [m] tower's radius at heights_tow
        
    # turbine's angles
    theta_cone = np.deg2rad(0)    # [rad] cone angle
    theta_yaw = np.deg2rad(0)     # [rad] yaw angle
    theta_pitch = np.deg2rad(0)   # [rad] shaft's pitch angle
    d_angle = np.deg2rad(6)       # [rad] angle step for the simulation
    
    # simulation time and time step
    dt = 0.15                    # [s] time step for the simulation
    TOTAL_TIME = 600               # [s] total time of the simulation
    time_steps = int(np.floor(TOTAL_TIME/dt))
    
    # atmosphere's variables
    shear_exponent = 0            # [-] velocity profile's shear exponent
    wind_speed = np.arange(V_IN,V_OUT+0.1,1)
    # wind_speed = [12]
    RHO = 1.225                   # [kg/m^3] density 


    # initial conditions
    initial_omega = np.array([6,6,6,6,6.426,7.229,8.032,8.836,9.6,\
                              9.6,9.6,9.6,9.6,9.6,9.6,9.6,9.6,9.6,\
                              9.6,9.6,9.6,9.6])    # [rad/s] initial value for rotational speed
    initial_omega = initial_omega*np.pi/30
    # initial_omega = np.ones(len(wind_speed))*0.5
    initial_pitch = np.deg2rad(np.array([0,0,0,0,0,0,0,0,4.502,7.266,9.292,\
                              10.958,12.499,13.896,15.2,16.432,\
                              17.618,18.758,19.86,20.927,21.963,\
                              22.975]))
    # optimal setting
    Cp_opt = 0.476               # [-] optimal power coefficient
    tsr_opt = (0.5*RHO*np.pi*R**5*rated_omega**3*Cp_opt/RATED_POWER)**(1/3) # [-] optimal tip speed ratio

    # results ashes 15 m/s with tower
    # file_ashes = pd.read_excel(r'../results/results_ashes_8ms_without_tower.xlsx')
    # file_ashes_time = np.array(file_ashes['Time'][2:-1])
    # file_ashes_pitch = np.array(file_ashes['Pitch angle'][2:-1])
    # file_ashes_power = np.array(file_ashes['Power (aero)'][2:-1])
    # file_ashes_omega = np.array(file_ashes['RPM'][2:-1])
    
    # boolean values
    local_a_calculation = True    
    # If I want to determine the axial induction factor locally for each
    # radial position I put True, otherwise to use a mean value from the entire
    # turbine I switch to False
    # For the third question, the pitch angle changes from 0 to 2 degrees for 
    # a certain period to see the reaction's delay in the power.
    stall_model = True
    # the dynamic stall model is activated
    adding_turbulence = True 
    # adding turbulence to the wind speed
    tower_shadow = False
    # considering tower's presence when the blades are spinning
    inertia_pitch = True
    # using Mann's turb files or Davis'
    Mann = False
    # printing results for this velocity
    print_velocity = [12,15,18,21]
    index = print_velocity-np.ones(len(print_velocity))*V_IN
    #%% tower height linear regression
    X = heights_tow.reshape(-1, 1)  # for linear regression it needs to be from 
    # highest to lowest
    y = a_x  # Dependent variable
    height_model = LinearRegression()
    height_model.fit(X, y)
    gain = height_model.coef_[0]      # slope
    offset = height_model.intercept_  # intercept
    
    #%% Opening file and saving the contents
    DATA = SavingData(number_of_airfoils = 6,path=aeroelasticity_dir) # saving data class
    DATA.opening_files()
    aoa,lift_coefficient,drag_coefficient,separation_function,linear_lift_coefficient,\
           stalled_lift_coefficient,radius = DATA.storing_data()
    thickness_to_chord = np.array((100, 60, 48, 36, 30.1, 24.1))
    
    #%% vector inizialization
    # blades
    position_blades = np.zeros((time_steps,B,3)) # position of the considered point in space
    azimuthal_angle_blade1 = np.zeros((time_steps,1)) #angle position of blade 1
    theta_blade = np.zeros((B,1)) #angle position of the three blades

    # velocity                                              
    V0_system1 = np.zeros((time_steps,B,3)) #velocity in ground system
    V0_system4 = np.zeros((time_steps,B,3)) #velocity in blade system
    W_induced_quasi_steady = np.zeros((time_steps,B,radius.shape[0],3)) #quasi-steady induced velocity
    W_induced_intermediate = np.zeros((time_steps,B,radius.shape[0],3)) #intermediate induced velocity
    U_turb = np.zeros((time_steps,B,radius.shape[0],3)) # turbulence velocity                                                                                                                              
    W_induced = np.zeros((time_steps,B,radius.shape[0],3)) #induced wind velocity
    w_induced_65m = np.zeros((time_steps)) # induced wind at 65.75 m along the blade
    w_turb_hub = np.zeros((len(wind_speed),time_steps)) # turb signal in 
    
    # time array and rotor steps for graph
    time_array = np.linspace(0, TOTAL_TIME, time_steps) #time array
    #rot_steps = time_array*omega/(2*np.pi)

    # loads (forces, thrust, and power)
    final_tangential_force = np.zeros((time_steps,radius.shape[0],B)) # p_t
    final_normal_force = np.zeros((time_steps,radius.shape[0],B)) # p_n
    torque = np.zeros((len(wind_speed),time_steps)) # aerodynamic torque over time
    power = np.zeros((len(wind_speed),time_steps)) # mechanical power over time
    thrust = np.zeros((time_steps)) # thrust over time
    thrust_first_blade = np.zeros((time_steps))
    thrust_second_blade = np.zeros((time_steps))
    thrust_third_blade = np.zeros((time_steps))
    pz_blade1_turb = np.zeros((time_steps)) # pz for the first blade with turbulence
    generator_torque = np.zeros((len(wind_speed),time_steps)) # generator torque
    power_el = np.zeros((len(wind_speed),time_steps))
    cp = np.zeros((len(wind_speed),time_steps))
        
    # axial and tangential induction factors
    a_values = np.zeros((time_steps,1)) #axial induction factor
    a_mean = 0 #initialization if a global induction factor is chosen

    # pitch angle
    pitch_angle_set = np.zeros((len(wind_speed),time_steps)) # setpoint pitch angle
    pitch_angle_i = np.zeros((len(wind_speed),time_steps)) # integral pitch angle
    pitch_angle_p = np.zeros((len(wind_speed),time_steps)) # proportional pitch angle
    TIP_PITCH = np.zeros((len(wind_speed),time_steps)) # [rad] tip pitch
    TIP_PITCH[:,0] = initial_pitch
    # angular velocity and tsr
    angular_velocity = np.zeros((len(wind_speed),time_steps)) #[rad/s] angular velocity
    angular_velocity[:,0] = initial_omega # setting the first two values to the initial one
    angular_velocity[:,1] = initial_omega # setting the first two values to the initial one
    # I am setting the first two because the rotational speed is calculated using
    # the value at the previous iteration. Hence, I need to start at the second 
    # position
    tsr = np.zeros((len(wind_speed),time_steps)) 
    
    # miscellaneous
    GK = np.zeros((len(wind_speed),time_steps))  # gain reduction for high pitch angles
    f_s = np.zeros((time_steps,B,radius.shape[0])) #separation function for dynamic wake

    #%% class initialization
    # create the transformation matrixes
    TRANSFORMATION_MATR = TransformationMatrixes(theta_cone,theta_yaw,theta_pitch)
    
    # define the position of each blade
    POSITION = PositionDefinition(TRANSFORMATION_MATR,H,L_s)
    
    # determine shear and velocity in system 4 from system 1
    UNDISTURBED_WIND = UndisturbedWindSpeed(TRANSFORMATION_MATR,H,shear_exponent)
    
    # determine induced wind on an element of a blade
    INDUCED_WIND = InducedWind(B,RHO,R)
    
    # loads calculation (lift, pz, py,...)
    LOADS_CALCULATION = LoadsCalculation(aoa, lift_coefficient,drag_coefficient,\
                             separation_function,linear_lift_coefficient,\
                             stalled_lift_coefficient,DATA.number_of_airfoils,\
                             thickness_to_chord,B,RATED_POWER,V_IN,V_OUT,RHO,\
                             theta_cone,theta_yaw,theta_pitch,
                             H,L_s,R,shear_exponent)
    
    # pitch and rotational speed calculation
    PITCH_CONTROLLER = PitchController(KK, Kp, Ki, w_ref, dt, max_change_pitch,\
                                       pitch_max, pitch_min, I_rot)
        
    
    
    #%% wind speed loop
    for mm in range(0,len(wind_speed)):
        t_begin = tm.perf_counter()
        ws_hub_height = wind_speed[mm]
        
        #%% turbulence creation
        if Mann:
            input_file_turb = r'..\..\..\Turbulence_generator\input.txt'
            turb_file_1 = os.path.join(
        '..', '..', '..', 'Turbulence_generator', f'ws{int(wind_speed[mm])}', 'sim1.bin')
            turb_file_2 = os.path.join(
        '..', '..', '..', 'Turbulence_generator', f'ws{int(wind_speed[mm])}', 'sim2.bin')
            turb_file_3 = os.path.join(
        '..', '..', '..', 'Turbulence_generator', f'ws{int(wind_speed[mm])}', 'sim3.bin')
            
            # saving the input values of turbulence creation
            input_data_turb = np.loadtxt(input_file_turb,dtype=str)
            ADDING_TURBULENCE = AddingTurbulence(dt, time_steps,input_data_turb,Mann)

            # saving the streamwise, normal and vertical turbulence. You need to remember
            # that in the simulation, the x and z directions are switched if compared with 
            # the one used here. This is why we use the first turb file for w turbulence
            w_turb_plane,x_turb,y_turb = ADDING_TURBULENCE.calculating_turbulence_field(turb_file_1)
            v_turb_plane,x_turb,y_turb = ADDING_TURBULENCE.calculating_turbulence_field(turb_file_2)
            u_turb_plane,x_turb,y_turb = ADDING_TURBULENCE.calculating_turbulence_field(turb_file_3)
            
        else:
            
            turb_file_turb_davis = np.load(os.path.join(
                '..', '..', '..', 'Turbulence_generator', f'ws{int(wind_speed[mm])}',
                f'NTM_V{int(wind_speed[mm])}_red.npz')) 
            file_path = os.path.join('..', '..', '..', 'Turbulence_generator', 
                                     f'ws{int(wind_speed[mm])}', 
                                     f'NTM_V{int(wind_speed[mm])}_params_red.yaml')
            
            # Apri il file e leggi il contenuto YAML
            with open(file_path, 'r') as file:
                input_data_turb = yaml.safe_load(file)

            ADDING_TURBULENCE = AddingTurbulence(dt, time_steps,input_data_turb,Mann)
            z_vec = turb_file_turb_davis['z_vec']
            w_turb_plane = turb_file_turb_davis['turb_u'] # streamwise
            u_turb_plane = turb_file_turb_davis['turb_w'] # vertical
            v_turb_plane = turb_file_turb_davis['turb_v'] 
            w_turb_plane = w_turb_plane.transpose(2, 0, 1)
            u_turb_plane = u_turb_plane.transpose(2, 0, 1)
            v_turb_plane = v_turb_plane.transpose(2, 0, 1)
            
            # saving te turbulence streamwise fluctuations at hub height 
            w_turb_hub[mm,:] = w_turb_plane[0:len(time_array),15,15]
        #%% time loop
        for ii in range(0,time_steps):
                # initialize p_n and p_t for each time step
                tangential_force = np.zeros((radius.shape[0],B))
                normal_force = np.zeros((radius.shape[0],B))
                
                # saving the current angular velocity. For the first and second iteration,
                # I will use the same value.
                if ii==0:
                    omega = angular_velocity[mm,0]
                else:
                    omega = angular_velocity[mm,ii-1]
                    
                # time update
                time = ii*dt
                    
                # saving the position of the blades at current time
                if ii==0:
                    theta_adding = 0
                else:
                    theta_adding = azimuthal_angle_blade1[ii-1]
                theta_blade[0],theta_blade[1],theta_blade[2] = POSITION.position_blade(omega,theta_adding,dt) 
                azimuthal_angle_blade1[ii] = theta_blade[0]
                
                # loop on the number of blades
                for jj in range(0,B):
                        theta_blade_considered = float(theta_blade[jj].item())
                        
                        # loop on every element of the blade
                        for kk in range(0,radius.shape[0]-1):
                            
                            # calculates the position of the element of the blade
                            position_blades[ii,jj,:] = POSITION.final_calculation_position\
                                             (theta_blade_considered, radius[kk])
                                             
                            # calculating the turbulence fluctuations on the element
                            if adding_turbulence: # I add the turbulence to the V0 in system 1
                                # interpolating the position of blade's element with
                                # the turbulence plane for each component
                                U_turb[ii,jj,kk,0] = ADDING_TURBULENCE.interpolating_turbulence\
                                                    (ii, position_blades[ii,jj,0],\
                                                    position_blades[ii,jj,1],u_turb_plane)
                                U_turb[ii,jj,kk,1] = ADDING_TURBULENCE.interpolating_turbulence\
                                                    (ii, position_blades[ii,jj,0],\
                                                    position_blades[ii,jj,1],v_turb_plane)
                                U_turb[ii,jj,kk,2] = ADDING_TURBULENCE.interpolating_turbulence\
                                                    (ii, position_blades[ii,jj,0],\
                                                    position_blades[ii,jj,1],w_turb_plane)
                                                        
                                # saving velocity in system 1 with turbuelence
                                V0_system1[ii,jj,:] = UNDISTURBED_WIND.velocity_system1(
                                                            position_blades[ii,jj,:],\
                                                            ws_hub_height,gain,offset,tower_shadow)\
                                                            +U_turb[ii,jj,kk,:]
                            else:
                                # saving velocity in system 1 without turbuelence
                                V0_system1[ii,jj,:] = UNDISTURBED_WIND.velocity_system1(
                                                            position_blades[ii,jj,:],\
                                                            ws_hub_height,gain,offset,tower_shadow)
                            
                            # saving velocity in system 4
                            V0_system4[ii,jj,:] = UNDISTURBED_WIND.velocity_system4(\
                                                            V0_system1[ii,jj,:])
                            
                            # airfoil data of the considered element
                            twist = np.deg2rad(DATA.blade_data[kk,1]) #twist angle
                            chord = DATA.blade_data[kk,2] # chord length
                            thick_to_chord = DATA.blade_data[kk,3] #ratio between thickness and chord
                            
                            # calculation relative velocity
                            # In the first iteration the induced wind is set to 0, as also
                            # the separation function. Hence, there will be a small
                            # unsteady period where it'll converge to the right value.
                            if ii == 0:
                                V_rel = LOADS_CALCULATION.relative_velocity(V0_system4\
                                                      [ii,jj,:], [0, 0, 0], radius[kk],\
                                                      omega)
                                W_z = 0 # z component of induced wind
                                
                                # loads calculation
                                lift,phi,normal_force[kk,jj],tangential_force[kk,jj],f_s[ii,jj,kk] = \
                                    LOADS_CALCULATION.calculation_loads(V_rel,twist, chord,\
                                                         thick_to_chord,ii,0,dt,time,stall_model,\
                                                         TIP_PITCH[mm,ii-1])
                            else:
                                V_rel = LOADS_CALCULATION.relative_velocity(V0_system4\
                                                      [ii,jj,:], W_induced[ii-1,jj,kk,:],\
                                                      radius[kk],omega)
                                W_z = W_induced[ii-1,jj,kk,2] # z component of induced wind
                                
                                # loads calculation
                                lift,phi,normal_force[kk,jj],tangential_force[kk,jj],f_s[ii,jj,kk] = \
                                    LOADS_CALCULATION.calculation_loads(V_rel,twist, chord,\
                                                         thick_to_chord,ii,f_s[ii-1,jj,kk],dt,time,\
                                                         stall_model,TIP_PITCH[mm,ii-1])
                                        
                            # induced wind calculation with dynamic wake
                            # tip loss correction
                            F = INDUCED_WIND.tip_loss_correction(radius[kk], phi)
                            # Glauert's correction and local axial induction factor
                            f_g,a = INDUCED_WIND.Glauert_correction(ii,W_z,V0_system4[ii,jj,:],\
                                                    a_mean,local_a_calculation)
                            # denominator of induced velocity's equation
                            induced_denominator = INDUCED_WIND.denominator_induced_wind\
                                                    (ii, V0_system4[ii,jj,:],f_g, W_z)
                            # quasi steady induced velocity (equilibrium with loads)
                            W_induced_quasi_steady[ii,jj,kk,:] = INDUCED_WIND.induced_wind_quasi_steady\
                                                    (ii,lift,phi, radius[kk], F, induced_denominator)
                            
                            # to not have a long transient, the first 5 iterations 
                            # have time constants very close to zero. Hence, the dynamic
                            # wake is applied from the 5th on.
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
                
                # save the p_n, p_t, torque, power, trust array for the current time step
                # values for r=65.75 m
                pz_blade1_turb[ii] = normal_force[8,0] # pz at 65.75 m for blade 1
                w_induced_65m[ii] = np.sqrt(W_induced[ii,0,8,0]**2+W_induced[ii,0,8,1]**2\
                                            +W_induced[ii,0,8,2]**2)
                
                # p_n,p_t,torque,mechanical power and cp
                final_tangential_force[ii,:,:] = tangential_force 
                final_normal_force[ii,:,:] = normal_force 
                torque[mm,ii] = np.trapezoid(radius*tangential_force[:,0],radius)+\
                             np.trapezoid(radius*tangential_force[:,1],radius)+\
                             np.trapezoid(radius*tangential_force[:,2],radius)
                power[mm,ii] = torque[mm,ii]*omega;
                thrust_first_blade[ii] =  np.trapezoid(normal_force[:,0],radius)
                thrust_second_blade[ii] =  np.trapezoid(normal_force[:,1],radius)
                thrust_third_blade[ii] =  np.trapezoid(normal_force[:,2],radius)
                thrust[ii] = thrust_first_blade[ii] +\
                             thrust_second_blade[ii]+\
                             thrust_third_blade[ii]
                # pitch controller valid from 2nd iteration
                if ii!=0:
                    # optimal cp region (below rated wind speed)
                    if omega<rated_omega:
                        
                        # M = K*omega**2
                        K = RATED_POWER/rated_omega**3
                        # K= 0.5*RHO*np.pi*R**5*Cp_opt/tsr_opt**3
                        
                        # generator torque 
                        generator_torque[mm,ii] = K*omega**2
                    
                    # pitching region
                    if omega>rated_omega:
                        # generator torque
                        generator_torque[mm,ii] = RATED_POWER/omega
                    
                    # setpoint pitch, integral and proportional pitch at current time
                    pitch_angle_set[mm,ii],pitch_angle_p[mm,ii],pitch_angle_i[mm,ii],GK[mm,ii] = PITCH_CONTROLLER.pitch_set_calculation(\
                                omega,pitch_angle_i[mm,ii-1], TIP_PITCH[mm,ii-1])
                    
                    # inserting inertia on pitch angle from the third iteration, 
                    # because it's a second order filter
                    if inertia_pitch:
                        if ii>1: 
                            pitch_angle = PITCH_CONTROLLER.inertia_pitch(pitch_angle_set[mm,ii-1], \
                                                     TIP_PITCH[mm,ii-1], TIP_PITCH[mm,ii-2])
                        else:
                            pitch_angle = pitch_angle_set[mm,ii]
                    else:
                        pitch_angle = pitch_angle_set[mm,ii]
                    # checking constraints on pitch value
                    TIP_PITCH[mm,ii] = PITCH_CONTROLLER.checking_pitch(TIP_PITCH[mm,ii-1], pitch_angle)
                    
                    # angular velocity update
                    angular_velocity[mm,ii] = PITCH_CONTROLLER.rotational_speed(torque[mm,ii-1], \
                                           generator_torque[mm,ii-1], omega)
                    
                    # electric power
                    power_el[mm,ii] = generator_torque[mm,ii]*angular_velocity[mm,ii]
                    tsr[mm,ii] = angular_velocity[mm,ii]*R/ws_hub_height
                    cp[mm,ii] = power[mm,ii]/(0.5*RHO*np.pi*R**2*ws_hub_height**3)
        # transforming pitch angles in degrees for better comprehension
        pitch_angle_p[mm,:] = np.rad2deg(pitch_angle_p[mm,:])
        pitch_angle_i[mm,:] = np.rad2deg(pitch_angle_i[mm,:])
        TIP_PITCH[mm,:] = np.rad2deg(TIP_PITCH[mm,:])
        print(f'The mean tip pitch for V={ws_hub_height} m/s is {np.mean(TIP_PITCH[mm,345:]):.3f}\n')
        print(f'The mean  tsr for V={ws_hub_height} m/s is {np.mean(tsr[mm,345:]):.3f}\n')
        t_end = tm.perf_counter()
        execution_time = t_end-t_begin
        print(f'V={wind_speed[mm]} m/s required {execution_time:.1f} s')
angular_velocity = angular_velocity*30/np.pi
#%% FIGURES
colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
mean_pitch = np.mean(TIP_PITCH[:,345:],axis=1)
mean_cp = np.mean(cp[:,345:],axis=1)
mean_tsr = np.mean(tsr[:,345:],axis=1)
mean_omega = np.mean(angular_velocity[:,345:],axis=1)
mean_power = np.mean(power[:,345:],axis=1)
# np.savez(r'../results/question1/results_davis_turbulence_off.npz',
#                  time_array=time_array, mean_pitch=mean_pitch,mean_cp=mean_cp,
#                  wind_speed=wind_speed, mech_power=power, ele_power=power_el,
#                  tip_pitch=TIP_PITCH,omega=angular_velocity,mean_tsr=mean_tsr)

# total pitch angle, with proportional and integral
# fig,ax = plt.subplots(1,1)
# # ax.plot(time_array,TIP_PITCH[0,:],linestyle='-',label='V=12 m/s',color = 'k')
# ax.plot(time_array,TIP_PITCH[int(index[0]),:],linestyle='-',label=f'V={print_velocity[0]} m/s',color = colors[0])
# ax.plot(time_array,TIP_PITCH[int(index[1]),:],linestyle='-',label=f'V={print_velocity[1]} m/s',color = colors[1])
# ax.plot(time_array,TIP_PITCH[int(index[2]),:],linestyle='-',label=f'V={print_velocity[2]} m/s',color = colors[2])
# ax.plot(time_array,TIP_PITCH[int(index[3]),:],linestyle='-',label=f'V={print_velocity[3]} m/s',color = colors[3])
# ax.set_xlabel(r't [s]')
# ax.set_ylabel(r'$\theta_p\:[^\circ]$')
# ax.set_xlim([time_array[10], time_array[-1]])
# ax.legend(loc='lower right')
# #ax.set_ylim([0.5,5])
# ax.minorticks_on()
# ax.grid()
# ax.tick_params(direction='in',right=True,top =True)
# ax.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
# ax.tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
# ax.tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)
# fig.savefig(r'../results/question2/pitch_above_rated_timeseries_turbulent.pdf')


# pitch angle for all wind speeds
# fig,ax = plt.subplots(1,1)
# ax.plot(wind_speed,mean_pitch,linestyle='-',color = 'k',label='u-BEM',marker='d')
# ax.plot(wind_speed,theta_p_report,linestyle='--',color = 'k',label='DTU 10MW report',marker='o')
# ax.set_xlabel(r'V [m/s]')
# ax.set_ylabel(r'$\theta_p\:[^\circ]$')
# ax.set_xlim([V_IN,V_OUT])
# # ax.set_ylim([0.4,0.52])
# ax.legend(loc="lower right", frameon=True)
# ax.minorticks_on()
# ax.grid()
# ax.tick_params(direction='in',right=True,top =True)
# ax.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
# ax.tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
# ax.tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)
# fig.savefig(r'../results/question2/pitch_different_ws_turbulent.pdf')

# power
# fig,ax = plt.subplots(1,1)
# # ax.plot(time_array,power[0,:],linestyle='-',label=f'Mech power for V=12 m/s',color = 'k')
# ax.plot(time_array,power[int(index[0]),:],linestyle='-',label=f'V={print_velocity[0]} m/s',color = colors[0])
# ax.plot(time_array,power[int(index[1]),:],linestyle='-',label=f'V={print_velocity[1]} m/s',color = colors[1])
# ax.plot(time_array,power[int(index[2]),:],linestyle='-',label=f'V={print_velocity[2]} m/s',color = colors[2])
# ax.plot(time_array,power[int(index[3]),:],linestyle='-',label=f'V={print_velocity[3]} m/s',color = colors[3])
# ax.set_xlabel(r't [s]')
# ax.set_ylabel(r'$P\:[W]$')
# ax.set_xlim([time_array[80], time_array[-1]])
# ax.legend(loc='upper right')
# ax.set_ylim([0.95e7,1.1e7])
# ax.minorticks_on()
# ax.grid()
# ax.tick_params(direction='in',right=True,top =True)
# ax.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
# ax.tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
# ax.tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)
# fig.savefig(r'../results/question1/power_ws_above_rated_without_turbulence.pdf')

# mechanical power for all wind speeds
# fig,ax = plt.subplots(1,1)
# ax.plot(wind_speed,mean_power*1e-3,linestyle='-',color = 'k',label='u-BEM',marker='d')
# ax.plot(wind_speed,P_mech_report,linestyle='--',color = 'k',label='DTU 10MW report',marker='o')
# ax.set_xlabel(r'V [m/s]')
# ax.set_ylabel(r'$P\:[kW]$')
# ax.set_xlim([V_IN,V_OUT])
# # ax.set_ylim([0.4,0.52])
# ax.legend(loc="lower right", frameon=True)
# ax.minorticks_on()
# ax.grid()
# ax.tick_params(direction='in',right=True,top =True)
# ax.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
# ax.tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
# ax.tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)
# fig.savefig(r'../results/question2/mechanical_power_different_ws_turbulent.pdf')


# # mechanical power and electric power
# fig,ax = plt.subplots(1,1)
# ax.plot(time_array,power[11,:]*1e-3,linestyle='-',label=f'mechanical power for V={wind_speed[11]} m/s',color = colors[0])
# ax.plot(time_array,power_el[11,:]*1e-3,linestyle='-',label=f'electric power for V={wind_speed[11]} m/s',color = colors[1])
# ax.legend(loc="lower right", frameon=True)
# ax.set_xlabel(r't [s]')
# ax.set_ylabel(r'$P\:[kW]$')
# ax.set_xlim([time_array[0], time_array[-1]])
# #ax.set_ylim([6e6,15e6])
# ax.minorticks_on()
# ax.grid()
# ax.tick_params(direction='in',right=True,top =True)
# ax.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
# ax.tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
# ax.tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)
# fig.savefig(r'../results/question2/power_mech_&_ele_ws15_turbulent.pdf')

# rotational speed for all wind speeds
# fig,ax = plt.subplots(1,1)
# ax.plot(wind_speed,mean_omega,linestyle='-',color = 'k',label='u-BEM',marker='d')
# ax.plot(wind_speed,rpm_report,linestyle='--',color = 'k',label='DTU 10MW report',marker='o')
# ax.set_xlabel(r'V [m/s]')
# ax.set_ylabel(r'$\omega\:[rpm]$')
# ax.set_xlim([V_IN,V_OUT])
# # ax.set_ylim([0.4,0.52])
# ax.legend(loc="lower right", frameon=True)
# ax.minorticks_on()
# ax.grid()
# ax.tick_params(direction='in',right=True,top =True)
# ax.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
# ax.tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
# ax.tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)
# fig.savefig(r'../results/question2/omega_different_ws_turbulent.pdf')


# power coefficient for all wind speeds below rated
# index_rated = int(np.floor(rated_velocity-V_IN))
# fig,ax = plt.subplots(1,1)
# # ax.plot(wind_speed[0:index_rated+1],cp[0:index_rated+1,-1],linestyle='-',color = 'k',label='$steady\:C_p$')
# ax.plot(wind_speed[0:index_rated+1],mean_cp[0:index_rated+1]*100,linestyle='-',color = 'k',label='u-BEM',marker='d')
# ax.plot(wind_speed[0:index_rated+1],C_p_report[0:index_rated+1]*100,linestyle='--',color = 'k',label='DTU 10MW report',marker='o')
# ax.axhline(y=Cp_opt*100,linestyle=':',color='k',label='optimal value')
# ax.set_xlabel(r'V [m/s]')
# ax.set_ylabel(r'$C_p\:[\%]$')
# ax.set_xlim([V_IN,wind_speed[index_rated]])
# # ax.set_ylim([0.4,0.52])
# ax.legend(loc="lower right", frameon=True)
# ax.minorticks_on()
# ax.grid()
# ax.tick_params(direction='in',right=True,top =True)
# ax.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
# ax.tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
# ax.tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)
# fig.savefig(r'../results/question2/cp_different_ws_below_rated_turbulent.pdf')

# # tsr for all wind speeds below rated
# fig,ax = plt.subplots(1,1)
# ax.plot(wind_speed[0:index_rated+1],mean_tsr[0:index_rated+1],linestyle='-',color = 'k',label='u-BEM',marker='d')
# ax.axhline(y=tsr_opt,linestyle='--',color='k',label='optimal tsr')
# ax.set_xlabel(r'V [m/s]')
# ax.set_ylabel(r'$\lambda$')
# ax.set_xlim([V_IN,wind_speed[index_rated]])
# #ax.set_ylim([0.4,0.52])
# ax.legend(loc="lower right", frameon=True)
# ax.minorticks_on()
# ax.grid()
# ax.tick_params(direction='in',right=True,top =True)
# ax.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
# ax.tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
# ax.tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)
# fig.savefig(r'../results/question2/tsr_different_ws_below_rated_turbulent.pdf')

# turbulent signal at 65.5 m for blade number 1 over time
# fig,ax = plt.subplots(1,1)
# ax.plot(time_array,w_turb_plane[0:len(time_array),15,15],linestyle='-',color = 'k')
# ax.set_xlabel(r't [s]')
# ax.set_ylabel(r'turbulence signal [m/s]')
# ax.set_xlim([time_array[0], time_array[-1]])
# ax.grid()
# ax.tick_params(direction='in',right=True,top =True)
# ax.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
# ax.tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
# ax.tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)
# fig.savefig(r'../results/question2/turbulent_fluctuations_25ms.pdf')

# fig,axs = plt.subplots(3,1,sharex=True)
# fig.subplots_adjust(hspace=0.3)
# axs[0].plot(time_array,w_turb_plane[0:len(time_array),15,15],linestyle='-',color = 'k')
# axs[1].plot(time_array,v_turb_plane[0:len(time_array),15,15],linestyle='-',color = 'k')
# axs[2].plot(time_array,u_turb_plane[0:len(time_array),15,15],linestyle='-',color = 'k')
# axs[0].set_title('Streamwise fluctuations')
# axs[1].set_title('Transverse fluctuations')
# axs[2].set_title('Vertical fluctuations')
# axs[2].set_xlabel(r'$t\: [s]$')
# axs[0].set_ylabel('V [m/s]')
# axs[1].set_ylabel('V [m/s]')
# axs[2].set_ylabel('V [m/s]')
# axs[2].set_xlim([time_array[0], time_array[-1]])
# axs[0].grid()
# axs[1].grid()
# axs[2].grid()
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
# fig.savefig(r'../results/question2/turbulent_fluctuations_25ms.pdf')
#%% calculating the execution time of the script

end_time = tm.perf_counter()
execution_time = end_time-start_time
print(f'The program required {execution_time:.1f} s')
