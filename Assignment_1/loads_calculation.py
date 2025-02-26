#%% IMPORT PACKAGES

import numpy as np
from scipy import interpolate
import pandas as pd
#%% INITIALIZE THE CLASS

class LoadsCalculation():
    def __init__(self,aoa, lift_coefficient,drag_coefficient,separation_function,
                             linear_lift_coefficient,stalled_lift_coefficient,   
                             number_of_airfoils,thickness_to_chord,B=3,P_rtd=10*1e6,V_in=4,V_out=25,RHO=1.225,TIP_PITCH=0,
                 omega=0.72,theta_cone=0,theta_yaw=0,theta_pitch=0,H=119,L_s=7.1,
                 R=89.15,shear_exponent=0):
        self.B = B                            # [-] Number of blades
        self.P_rtd = P_rtd                    # [W] Turbine power
        self.V_in = V_in                      # [m/s] Cut-in speed
        self.V_out = V_out                    # [m/s] Cut-out speed
        self.RHO = RHO                        # [kg/m^3] Density
        self.TIP_PITCH =  TIP_PITCH           # [rad] pitch of the tip
        self.omega = omega                    # [rad/s] Rotational speed
        self.theta_cone = theta_cone          # [rad] Cone angle
        self.theta_yaw = theta_yaw            # [rad] Yaw angle
        self.theta_pitch = theta_pitch        # [rad] Shaft pitch angle 
        self.H = H                            # [m] Hub height
        self.L_s = L_s                        # [m] Nacelle length
        self.R = R                            # [m] Rotor radius
        self.shear_exponent = shear_exponent  # [-] Velocity shear exponent
        self.number_of_airfoils = number_of_airfoils
        self.aoa = aoa[:,0]
        self.thickness_to_chord = thickness_to_chord
        self.lift_coefficient = lift_coefficient
        self.drag_coefficient = drag_coefficient
        self.separation_function = separation_function
        self.linear_lift_coefficient = linear_lift_coefficient
        self.stalled_lift_coefficient = stalled_lift_coefficient
        # self.interp_cd = interpolate.interp1d(aoa[:,0], drag_coefficient,axis=0,
        #                                    kind='linear')
        # self.interp_fs = interpolate.interp1d(aoa[:,0], separation_function,axis=0,
        #                                    kind='linear')
        # self.interp_linear_cl = interpolate.interp1d(aoa[:,0], linear_lift_coefficient,axis=0,
        #                                    kind='linear')
        # self.interp_stalled_cl = interpolate.interp1d(aoa[:,0], stalled_lift_coefficient,axis=0,
        #                                    kind='linear')
    def relative_velocity(self,vel_sys4,induced_wind,radial_position):
        '''This function determines the realtive velocity on the element chosen
        on the blade in system 4 (blade).
        
        Parameters:
            vel_sys4 (1-D array like): 
                wind speed in blade system.
            induced_wind (1-D array like): 
                induced wind in blade system.
            radial_position : 
                radial position on the blade of the considered element.

        Returns:
            V_rel (1-D array like):
                relative velocity at the radial position.

        '''
        V_rel_y = vel_sys4[1]+induced_wind[1]-self.omega*radial_position\
            *np.cos(self.theta_cone)
        V_rel_z = vel_sys4[2]+induced_wind[2]
        return np.array([0, V_rel_y, V_rel_z])
    
    def calculation_loads(self,V_rel,twist,c,thick_to_chord,iteration,fs_bef,dt,t,\
                          stall_model):
        '''This function calculates the lift, drag, tangential force, and normal
        force for each element on the blade. The calculation is done by interpolating
        the angle of attack between the data given in the uploaded files.
        
        Parameters:
        V_rel (1-D array like):
            relative velocity on the element
        twist (float):
            local twist angle of the element.
        c (float): 
            chord length of the element
        thick_to_chord (float):
            ratio between thickness and chord in percentage
        aoa (2-D array like): 
            matrix containing each angle of attack for which C_l and C_d are
            calcualted, for each airfoil.
        lift_coefficient (2-D array like): 
            matrix containing the lift coefficient for each angle of attack for
            each airfoil
        drag_coefficient (2-D array like): 
            matrix containing the drag coefficient for each angle of attack for
            each airfoil
        separation_function (2-D array like): 
            matrix containing the separation function for the dynamic wake model
            for each angle of attack for each airfoil
        linear_lift_coefficient (2-D array like): 
            matrix containing the linear lift coefficient for each angle of attack 
            for each airfoil
        stalled_lift_coefficient (2-D array like): 
            matrix containing the stalled lift coefficient for each angle of attack 
            for each airfoil
        number_of_airfoils (int): number of used airfoils
        thickness_to_chord (1-D array like):
            array containing the thickness to chord ratio (in percentage) of the 
            used airfoils.
        iteration (int): 
            time loop iteration
        fs_bef (float): 
            separation function's value at the previous time step
        dt (float): 
            time step of the simulation
        t (float):
            current time of the simulation
        stall_model:
            boolean. If True, the stall model is used, otherwhise the static
            model is used

        Returns:
        lift (float):
            lift per unit length for the blade's element
        phi (float): 
            flow angle for the blade's element
        infinitesimal_norm_force (float): 
            p_n for the blade's element
        infinitesimal_tang_force (float): 
            p_t for the blade's element
        fs_now (float):
            separation function at the current time step
        '''
        abs_V_rel = np.sqrt(V_rel[1]**2+V_rel[2]**2) #module of the relative velocity
        phi = np.arctan2(V_rel[2],-V_rel[1]) #flow angle
        local_pitch = twist+self.TIP_PITCH[iteration] 
        angle_attack = phi-local_pitch
        # interpolating with scipy
        #clthick = np.zeros((number_of_airfoils,1)) #lift coefficient
        # cdthick = self.interp_cd(np.rad2deg(angle_attack)) #drag coefficient
        # fs_thick = self.interp_fs(np.rad2deg(angle_attack)) #separation function
        # linear_cl_thick = self.interp_linear_cl(np.rad2deg(angle_attack))#linear lift coefficient
        # stalled_cl_thick = self.interp_stalled_cl(np.rad2deg(angle_attack)) #stalled lift coefficient
        
        #interpolating with numpy
        clthick = np.empty(self.number_of_airfoils)
        cdthick = np.empty(self.number_of_airfoils)
        fs_thick = np.empty(self.number_of_airfoils)
        linear_cl_thick  = np.empty(self.number_of_airfoils)
        stalled_cl_thick = np.empty(self.number_of_airfoils)
        for ii in range(self.number_of_airfoils):
            clthick[ii] = np.interp(np.rad2deg(angle_attack),self.aoa, self.lift_coefficient[:,ii])
            cdthick[ii] = np.interp(np.rad2deg(angle_attack),self.aoa, self.drag_coefficient[:,ii])
            fs_thick[ii] = np.interp(np.rad2deg(angle_attack),self.aoa, self.separation_function[:,ii])
            linear_cl_thick[ii] = np.interp(np.rad2deg(angle_attack),self.aoa, self.linear_lift_coefficient[:,ii])
            stalled_cl_thick[ii] = np.interp(np.rad2deg(angle_attack),self.aoa, self.stalled_lift_coefficient[:,ii])
        # In the interpolation function, I have to reverse the arrays because
        # np.intepr wants the array to be with increasing values
        clift_static = np.interp(thick_to_chord,self.thickness_to_chord[-1::-1],clthick[-1::-1])
        cdrag = np.interp(thick_to_chord,self.thickness_to_chord[-1::-1],cdthick[-1::-1])
        fs_stat = np.interp(thick_to_chord,self.thickness_to_chord[-1::-1],fs_thick[-1::-1])
        linear_clift = np.interp(thick_to_chord,self.thickness_to_chord[-1::-1],linear_cl_thick[-1::-1])
        stalled_clift = np.interp(thick_to_chord,self.thickness_to_chord[-1::-1],stalled_cl_thick[-1::-1])
        
        if stall_model:
            tau = 4*c/abs_V_rel
            fs_now = fs_stat+(fs_bef-fs_stat)*np.exp(-dt/tau)
            clift_dyn_stall = fs_now*linear_clift+(1-fs_now)*stalled_clift
            lift = 1/2*self.RHO*c*clift_dyn_stall*abs_V_rel**2
        else:
            fs_now = fs_stat
            lift = 1/2*self.RHO*c*clift_static*abs_V_rel**2
        # determining the loads on the blade's element
        drag = 1/2*self.RHO*c*cdrag*abs_V_rel**2
        infinitesimal_tang_force = lift*np.sin(phi)-drag*np.cos(phi)
        infinitesimal_norm_force = lift*np.cos(phi)+drag*np.sin(phi)
              
        return lift,phi,infinitesimal_norm_force,infinitesimal_tang_force,fs_now
    
