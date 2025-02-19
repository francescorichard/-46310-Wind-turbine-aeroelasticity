'''
This file is an attempt to do Aeroelasticity's Assignment 1 with the implementation
of classes.
'''

#%% IMPORT PACKAGES

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib as mpl
import time as tm
#%% plot commands
start_time = tm.perf_counter()

#size
mpl.rcParams['figure.figsize'] = (12,8)

#font size of label, title, and legend
mpl.rcParams['font.size'] = 25
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.labelsize'] = 25
mpl.rcParams['axes.titlesize'] = 30
mpl.rcParams['legend.fontsize'] = 25

#Lines and markers
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 7
mpl.rcParams['scatter.marker'] = "+"
plt_marker = "d"

#Latex font
plt.rcParams['font.family'] = 'serif'  # Simula il font di LaTeX
plt.rcParams['mathtext.fontset'] = 'cm'  # Usa Computer Modern per la matematica

#Export
mpl.rcParams['savefig.bbox'] = "tight"
#%% INITIALIZE THE CLASS

class QuasiSteady():
    def __init__(self,B=3,P_rtd=10*1e6,V_in=4,V_out=25,RHO=1.225,TIP_PITCH=0,
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
    
    
    def matrix_a1(self):
         '''Transformation matrix from ground system (system 1) to the nacelle
         considering the rotation of the latter due to yaw alignment.
         
         Parameter:
             theta_yaw (float) :
                 yaw angle of the turbine
        
         Return:
            a_1 (2-D array-like):
                transformation matrix
         '''
         a_1 = np.array([[1, 0, 0],
                         [0, np.cos(self.theta_yaw), np.sin(self.theta_yaw)],
                         [0, -np.sin(self.theta_yaw), np.cos(self.theta_yaw)]])
         return a_1
     
    def matrix_a2(self):
         '''Transformation matrix from ground system (system 1) to the nacelle
         considering the rotation of the latter due to pitch.
         
         Parameter:
             theta_pitch (float) :
                 pitch angle of the shaft
        
         Return:
            a_2 (2-D array-like):
                transformation matrix
         '''
         a_2 = np.array([[np.cos(self.theta_pitch), 0, -np.sin(self.theta_pitch)],
                         [0, 1, 0],
                         [np.sin(self.theta_pitch), 0, np.cos(self.theta_pitch)]])
         return a_2
     
    def matrix_a12(self):
        '''Final transformation matrix from ground system (system 1) to the nacelle
         considering all the variations in the coordinate system.
         
        Parameter:
             a_1 (2-D array-like):
                 transformation matrix considering yaw alignment
             a_2 (2-D array-like):
                 transformation matrix considering nacelle pitch
             a_3 (2-D array-like):
                 transformation matrix considering roll 
        
        Return:
            a_12 (2-D array-like):
                transformation matrix from ground to nacelle
            a_21 (2-D array-like):
                transformation matrix from nacelle to ground
        '''
        a_3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) #roll not considered
        a_12 = (a_3 @ self.a_2) @ self.a_1
        a_21 = a_12.transpose()
        return a_12,a_21

    def matrix_a23(self,theta_blade):
        '''Transformation matrix from nacelle system (system 2) to the shaft
        (system 3) considering the rotation of the latter due to rotation of the blades.
        You can imagine this coordinate system positioned at the very root of 
        the blade.
        
        Parameter:
            theta_blade (float):
                azimuthal angle of the blade considered
       
        Return:
           a_23 (2-D array-like):
               transformation matrix from nacelle to shaft
        '''
        a_23 = np.array([[np.cos(theta_blade), np.sin(theta_blade), 0],
                        [-np.sin(theta_blade), np.cos(theta_blade), 0],
                        [0, 0, 1]])
        return a_23
     
    def matrix_a34(self):
        '''Transformation matrix from shaft system (system 3) to the blade 
        (system 4)considering the misalignment with the root due to the cone
        angle.
        
        Parameter:
            theta_cone (float):
                cone angle of the blade considered
       
        Return:
           a_34 (2-D array-like):
               transformation matrix from shaft to blade
        '''
        a_34 = np.array([[np.cos(self.theta_cone), 0, -np.sin(self.theta_cone)],
                        [0, 1, 0],
                        [np.sin(self.theta_cone), 0, np.cos(self.theta_cone)]])
        return a_34

    def matrix_a14(self):
        '''Final transformation matrix from ground system (system 1) to the
        blade.
         
        Parameter:
             a_12 (2-D array-like):
                 transformation matrix from ground to nacelle
             a_23 (2-D array-like):
                 transformation matrix from nacelle to shaft
             a_34 (2-D array-like):
                 transformation matrix from shaft to blade 
        
        Return:
            a_14 (2-D array-like):
                transformation matrix from ground to blade
            a_41 (2-D array-like):
                transformation matrix from blade to ground
        '''
        a_14 = self.a_34 @ (self.a_23 @ self.a_12)
        a_41 = a_14.T
        return a_14,a_41
    
    def position_point_system1(self,x_b):
        '''Position of the point on the blade from system 1 (ground).
        The final position is the sum of the radial positions of the three 
        systems: ground, nacelle, and blade.
         
        Parameter:
           r_groud (1-D array-like):
               position of the nacelle from system 1
           r_nacelle (1-D array-like):
               position of the shaft from system 1
           r_blade (1-D array-like):
               position of the point on the blade from system 1
        
        Return:
           r_tot (1-D array-like):
               position of the point on the blade from system 1
        '''
        r_ground = np.array([[self.H],[0], [0]])
        r_nacelle = self.a_21 @ np.array([[0],[0], [-self.L_s]])
        r_blade = self.a_41 @ np.array([[x_b],[0], [0]])
        r_final = r_ground + r_nacelle + r_blade
        return r_final.flatten()
    
    def position_blade(self,t):
        '''This function determines the azimuth position of the blades at 
        time instant t.
        
        Parameter:
            omega (float):
                angular velocity of the blades
            t (int):
                time position
        
        Return:
            theta_blade_1 (float):
                azimuthal position of blade 1 at time t
            theta_blade_2 (float):
                azimuthal position of blade 1 at time t
            theta_blade_3 (float):
                azimuthal position of blade 1 at time t
        '''
        theta_blade_1 = self.omega*t
        theta_blade_2 = theta_blade_1+2/3*np.pi
        theta_blade_3 = theta_blade_1+4/3*np.pi
        return theta_blade_1,theta_blade_2,theta_blade_3
    
    def final_calculation_position(self,theta_blade,x_b):
         '''This function determines the position of the point on the blade 
         from system 1 (ground).
         
         Parameter:
             theta_blade (float): azimuth position of the blades at time 
             instant t
             x_b (float):
                 position on the blade from blade system
        
         Return:
             r_point (float):
                 position of the point on the blade from system 1
                 
         '''
         self.a_1 = QuasiSteady.matrix_a1(self)
         self.a_2 = QuasiSteady.matrix_a2(self)
         self.a_12, self.a_21 = QuasiSteady.matrix_a12(self)
         self.a_23 = QuasiSteady.matrix_a23(self,theta_blade)
         self.a_34 = QuasiSteady.matrix_a34(self)
         self.a_14, self.a_41 = QuasiSteady.matrix_a14(self)
         r_point = QuasiSteady.position_point_system1(self,x_b)
         return r_point,self.a_1,self.a_2,self.a_12,self.a_23,self.a_34,\
             self.a_14,self.a_21,self.a_41
    
    def velocity_system1(self,x,ws_hub_height,a_x):
        '''This function determines the velocity of the point on the blade 
        from system 1 (ground).
        
        Parameter:
            x (float): 
                position of the point ins system 1
            ws_hub_height (float):
                wind speed at hub height
            a_x = radius of the tower at the point height
            
        Return:
            vel_sys1 (1-D array like): 
                velocity of the point in system 1 (ground)
        '''
        v0_x = ws_hub_height*(x[0]/self.H)**self.shear_exponent
        if x[0] > self.H:
            velocity_sys1 = np.array([0, 0, v0_x])
        else:
            r = np.sqrt(x[1]**2+x[2]**2)
            v_r = x[2]/r*v0_x*(1-(a_x/r)**2)
            v_theta = x[1]/r*v0_x*(1+(a_x/r)**2)
            v_y= x[1]/r*v_r-x[2]/r*v_theta
            v_z = x[2]/r*v_r+x[1]/r*v_theta
            velocity_sys1 = np.array([0, v_y, v_z])
        return velocity_sys1
    
    def velocity_system4(self,vel_sys1):
        '''This function determines the velocity of the point on the blade 
        from system 4 (blade).
        
        Parameter:
            vel_sys1 (1-D array like): 
                velocity of the point in system 1 (ground)
       
        Return:
            vel_sys4 (1-D array like): 
                velocity of the point in system 4 (blade)
        '''
        vel_sys4 = self.a_14 @ vel_sys1
        return vel_sys4
    
    def dynamic_wake(self,W_qs_now,W_qs_bef,W_int_bef,W_ind_bef,tau_1,tau_2,dt):
        k = 0.6
        H = W_qs_now + k*tau_1*(W_qs_now-W_qs_bef)/dt
        W_int_now = H+(W_int_bef-H)*np.exp(-dt/tau_1)
        W_ind_now = W_int_now+(W_ind_bef-W_int_now)*np.exp(-dt/tau_2)
        return W_ind_now,W_int_now
    
    def time_constants_induced_wind(self,a,V_0,r):
        a = np.min(np.array([a,0.5]))
        tau_1 = 1.1/(1-1.3*a)*self.R/V_0
        tau_2 = (0.39-0.26*(r/self.R)**2)*tau_1
        return tau_1,tau_2
    
    def induced_wind_quasi_steady(self,iteration,lift,phi,radial_pos,F,denominator):
        '''Calculates the induced wind in the considered element of the blade.
        
        Parameters:
            iteration (int):
                time step.
            lift (float):
                lift per chord length in the considered point on the blade.
            phi (float):
                flow angle.
            radial_pos (float): 
                radial position on the blade of the considered element.
            F (float):
                Prandtl's tip loss factor.
            denominator (float):
                module of |V_0+f_g*W_n|.

        Returns:
            W_y (float):
                induced wind y-component.
            W_z (float):
                induced wind z-component.
        '''
        if iteration == 0:
            W_z = 0
            W_y = 0
        else:
            W_z = -self.B*lift*np.cos(phi)/(4*np.pi*self.RHO*radial_pos*F*denominator)
            W_y = -self.B*lift*np.sin(phi)/(4*np.pi*self.RHO*radial_pos*F*denominator)
        return np.array([0, W_y, W_z])
    
    @staticmethod
    def denominator_induced_wind(iteration,vel_sys4,f_g,W_z):
        '''Calculates the induced wind in the considered element of the blade.
        
        Parameters:
            iteration (int):
                time step.
            vel_sys4 (1-D array like): 
                velocity of the point in system 4 (blade)
            phi (float):
                flow angle.
            f_g (float): 
                Glauert's correction
            W_z (float):
                induced wind z-component.
            denominator (float):
                module of |V_0+f_g*W_n|.

        Returns:
            denominator (float):
                module of |V_0+f_g*W_n|.
        '''
        if iteration == 0:
            W_z = 0
        denominator = np.sqrt(vel_sys4[1]**2+(vel_sys4[2]+f_g*W_z)**2)
        return denominator
    
    @staticmethod
    def Glauert_correction(iteration,W_z,vel_sys4,aa_mean,local_calculation):
        '''It calculates the Glauert's correction based on the value of the 
        axial induction factor.
        
        Parameters:
            iteration (int):
                time step
            W_z (float):
                induced wind z-component.
            vel_sys4 (1-D array like): 
                velocity of the point in system 4 (blade)
            aa (float):
                axial induction factor
            local_calc (Boolean):
                usage of local induction factor (True) or turbine mean value 
                
        Returns:
            f_g (float):
                Glauert's correction'
        '''
        if local_a_calculation:
            if iteration == 0:
                W_z = 0
            V_0 = np.sqrt(vel_sys4[1]**2+vel_sys4[2]**2)
            aa = -W_z/V_0
        else:
            aa = aa_mean
            
        if aa <= 1/3:
            f_g = 1
        else:
            f_g = 1/4*(5-3*aa)
        return f_g,aa 

    def tip_loss_correction(self,r,phi):
        '''It calculates the tip loss correction from the hypothesis of
        infinite blades.
        
        Parameters:
            r (float):
                radial position on the blade of the considered element.
            phi (float):
                flow angle

                
        Returns:
            F (float):
                Glauert's correction'
        '''
        ff = np.divide(self.B*(self.R-r),2*r*np.sin(np.abs(phi)))
        F = 2/np.pi*np.arccos(np.exp(-ff))
        return F
    
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
    
    def convergence_a_aPrime(self,V_rel,twist,c,thick_to_chord,aoa,
                             lift_coefficient,drag_coefficient,separation_function,
                             linear_lift_coefficient,stalled_lift_coefficient,   
                             number_of_airfoils,thickness_to_chord,iteration,fs_bef,dt,t):

        abs_V_rel = np.sqrt(V_rel[1]**2+V_rel[2]**2)
        phi = np.arctan2(V_rel[2],-V_rel[1])
        local_pitch = twist+self.TIP_PITCH[iteration]
        angle_attack = phi-local_pitch
        # interpolating
        #clthick = np.zeros((number_of_airfoils,1))
        cdthick = np.zeros((number_of_airfoils,1))
        fs_thick = np.zeros((number_of_airfoils,1))
        linear_cl_thick = np.zeros((number_of_airfoils,1))
        stalled_cl_thick = np.zeros((number_of_airfoils,1))
        for kk in range(0,number_of_airfoils):
            #clthick[kk] = interpolate.interp1d(aoa[:,kk], lift_coefficient[:,kk],
            #                                   kind='linear')(np.rad2deg(angle_attack))
            cdthick[kk] = interpolate.interp1d(aoa[:,kk], drag_coefficient[:,kk],
                                               kind='linear')(np.rad2deg(angle_attack))
            fs_thick[kk] = interpolate.interp1d(aoa[:,kk], separation_function[:,kk],
                                               kind='linear')(np.rad2deg(angle_attack))
            linear_cl_thick[kk] = interpolate.interp1d(aoa[:,kk], linear_lift_coefficient[:,kk],
                                               kind='linear')(np.rad2deg(angle_attack))
            stalled_cl_thick[kk] = interpolate.interp1d(aoa[:,kk], stalled_lift_coefficient[:,kk],
                                               kind='linear')(np.rad2deg(angle_attack))
        #clift = interpolate.interp1d(thickness_to_chord,clthick[:,0], 
        #                             kind='linear')(thick_to_chord)
        cdrag = interpolate.interp1d(thickness_to_chord,cdthick[:,0], 
                                     kind='linear')(thick_to_chord)
        fs_stat = interpolate.interp1d(thickness_to_chord,fs_thick[:,0], 
                                     kind='linear')(thick_to_chord)
        linear_clift = interpolate.interp1d(thickness_to_chord,linear_cl_thick[:,0], 
                                     kind='linear')(thick_to_chord)
        stalled_clift = interpolate.interp1d(thickness_to_chord,stalled_cl_thick[:,0], 
                                     kind='linear')(thick_to_chord)
        # calculate dynamic stall
        tau = 4*c/abs_V_rel
        fs_now = fs_stat+(fs_bef-fs_stat)*np.exp(-dt/tau)
        
        clift_dyn_stall = fs_now*linear_clift+(1-fs_now)*stalled_clift
        lift = 1/2*self.RHO*c*clift_dyn_stall*abs_V_rel**2
        drag = 1/2*self.RHO*c*cdrag*abs_V_rel**2
        infinitesimal_tang_force = lift*np.sin(phi)-drag*np.cos(phi)
        infinitesimal_norm_force = lift*np.cos(phi)+drag*np.sin(phi)
            
        
        return lift,phi,infinitesimal_norm_force,infinitesimal_tang_force,fs_now
    
    
#%% MAIN
if __name__ == "__main__":
    #%% Initialize values

    B = 3                         # [-] number of blades
    RATED_POWER = 10*1e6          # [MW] rated power
    V_IN = 4                      # [m/s] cut in speed 
    V_OUT = 25                    # [m/s] cut-out speed
    RHO = 1.225                   # [kg/m^3] density 
    omega = 0.72                  # [rad/s] rotational speed 
    theta_cone = np.deg2rad(0)    # [rad] cone angle
    theta_yaw = np.deg2rad(0)     # [rad] yaw angle
    theta_pitch = np.deg2rad(0)   # [rad] shaft's pitch angle
    H =119                        # [m] hub height
    L_s = 7.1                     # [m] shaft length 
    R = 89.15                     # [m] turbine's radius
    d_angle = np.deg2rad(6)       # [rad] angle step for the simulation
    dt = d_angle/omega            # [s] time step for the simulation
    TOTAL_TIME = 180              # [s] total time of the simulation
    time_steps = int(np.floor(TOTAL_TIME/dt))
    shear_exponent = 0            # [-] velocity profile's shear exponent
    ws_hub_height = 8             # [m/s] hub height wind speed
    a_x = 3.32                    # [m] tower's radius
    local_a_calculation = True    # local calculation of axial induction factor
    TIP_PITCH = np.zeros(time_steps)  # [rad] tip pitch
    third_point = True
    if third_point:
        TIP_PITCH[(np.arange(time_steps)*dt>=100) & (np.arange(time_steps)*dt <= 150)] = np.deg2rad(2)
    #%% Opening file and saving the contents
    cylinder = np.loadtxt('C:\\COPENAGHEN PRIMO ANNO\\AEROELASTICITY\\turbine_data\\cylinder_ds.txt')
    FFA_W3_301 = np.loadtxt('C:\\COPENAGHEN PRIMO ANNO\\AEROELASTICITY\\turbine_data\\FFA-W3-301_ds.txt');
    FFA_W3_360 = np.loadtxt('C:\\COPENAGHEN PRIMO ANNO\\AEROELASTICITY\\turbine_data\\FFA-W3-360_ds.txt');
    FFA_W3_480 = np.loadtxt('C:\\COPENAGHEN PRIMO ANNO\\AEROELASTICITY\\turbine_data\\FFA-W3-480_ds.txt');
    FFA_W3_600 = np.loadtxt('C:\\COPENAGHEN PRIMO ANNO\\AEROELASTICITY\\turbine_data\\FFA-W3-600_ds.txt');
    FFA_W3_2411 = np.loadtxt('C:\\COPENAGHEN PRIMO ANNO\\AEROELASTICITY\\turbine_data\\FFA-W3-241_ds.txt');
    blade_data = np.loadtxt('C:\\COPENAGHEN PRIMO ANNO\\AEROELASTICITY\\turbine_data\\bladedat.txt');
    number_of_airfoils = 6;
    radius = blade_data[:,0]
    # save data in different matrix.I want to save a matrix with every variable.
    # Angle of attack, lift coefficient, drag coefficient, and thrust coefficient
    total_data = np.concatenate((cylinder,
                                FFA_W3_600,
                                FFA_W3_480,
                                FFA_W3_360,
                                FFA_W3_301,
                                FFA_W3_2411),axis=1)
    
    #initialize the lift, drag, angle of attack, and thickness to chord
    #ratio matrixes
    thickness_to_chord = np.array((100, 60, 48, 36, 30.1, 24.1))
    aoa = np.zeros((total_data.shape[0],number_of_airfoils))
    lift_coefficient = np.zeros((total_data.shape[0],number_of_airfoils))
    drag_coefficient = np.zeros((total_data.shape[0],number_of_airfoils))
    separation_function = np.zeros((total_data.shape[0],number_of_airfoils))
    linear_lift_coefficient = np.zeros((total_data.shape[0],number_of_airfoils))
    stalled_lift_coefficient = np.zeros((total_data.shape[0],number_of_airfoils))
    position = 0
    for kk in range(0,number_of_airfoils):
        aoa[:,kk] = total_data[:,position]
        lift_coefficient[:,kk] = total_data[:,position+1]
        drag_coefficient[:,kk] = total_data[:,position+2]
        separation_function[:,kk] = total_data[:,position+4]
        linear_lift_coefficient[:,kk] = total_data[:,position+5]
        stalled_lift_coefficient[:,kk] = total_data[:,position+6]
        position += 7


    #%% vector inizialization
    position_blades = np.zeros((time_steps,B,3))
    V0_system1 = np.zeros((time_steps,B,3))
    V0_system4 = np.zeros((time_steps,B,3)) 
    W_induced_quasi_steady = np.zeros((time_steps,B,radius.shape[0],3))
    W_induced_intermediate = np.zeros((time_steps,B,radius.shape[0],3))
    W_induced = np.zeros((time_steps,B,radius.shape[0],3))
    f_s = np.zeros((time_steps,B,radius.shape[0]))
    azimuthal_angle_blade1 = np.zeros((time_steps,1))
    time_array = np.linspace(0, TOTAL_TIME, time_steps)
    theta_blade = np.zeros((B,1))
    final_tangential_force = np.zeros((time_steps,radius.shape[0],B))
    final_normal_force = np.zeros((time_steps,radius.shape[0],B))
    torque = np.zeros((time_steps,1))
    power = np.zeros((time_steps,1))
    thrust = np.zeros((time_steps,1))
    
    #initialization of axial and tangential induction factors
    a_values = np.zeros((time_steps,1))
    a_mean = 0
    
    #%% resolution
    UNSTEADY_SOLVER = QuasiSteady(B,RATED_POWER,V_IN,V_OUT,RHO,TIP_PITCH,
                                  omega,theta_cone,theta_yaw,theta_pitch,
                                  H,L_s,R,shear_exponent)
    
#time loop
for ii in range(0,time_steps):
        #initialize p_n and p_t for each time step
        tangential_force = np.zeros((radius.shape[0],B))
        normal_force = np.zeros((radius.shape[0],B))
        
        #time update
        time = ii*dt 
        
        #saving the position of the blades at current time
        theta_blade[0],theta_blade[1],theta_blade[2] = UNSTEADY_SOLVER.\
            position_blade(time) 
        azimuthal_angle_blade1[ii] = theta_blade[0]
        
        #loop on the number of blades
        for jj in range(0,B):
                theta_blade_considered = float(theta_blade[jj])
                
                #loop on every element of the blade
                for kk in range(0,radius.shape[0]-1):
                    
                    #calculates the velocity in system 1 and 4
                    position_blades[ii,jj,:],a_1,a_2,a_12,a_23,a_34,a_14,a_21,\
                              a_41 = UNSTEADY_SOLVER.final_calculation_position\
                                     (theta_blade_considered, radius[kk])
                    V0_system1[ii,jj,:] = UNSTEADY_SOLVER.velocity_system1(
                                                    position_blades[ii,jj,:],\
                                                    ws_hub_height,a_x)
                    V0_system4[ii,jj,:] = UNSTEADY_SOLVER.velocity_system4(\
                                                    V0_system1[ii,jj,:])
                    
                    #data of the considered element
                    twist = np.deg2rad(blade_data[kk,1])
                    chord = blade_data[kk,2]
                    thick_to_chord = blade_data[kk,3]
                    
                    #calculation relative velocity
                    if ii == 0:
                        V_rel = UNSTEADY_SOLVER.relative_velocity(V0_system4\
                                              [ii,jj,:], [0, 0, 0], radius[kk])
                        W_z = 0
                        lift,phi,normal_force[kk,jj],tangential_force[kk,jj],f_s[ii,jj,kk] = \
                            UNSTEADY_SOLVER.convergence_a_aPrime(V_rel,twist, chord,\
                                                 thick_to_chord, aoa, lift_coefficient,\
                                                 drag_coefficient,separation_function,
                                                 linear_lift_coefficient,stalled_lift_coefficient,\
                                                 number_of_airfoils,\
                                                 thickness_to_chord,ii,0,dt,time)
                    else:
                        V_rel = UNSTEADY_SOLVER.relative_velocity(V0_system4\
                                              [ii,jj,:], W_induced[ii-1,jj,kk,:],\
                                              radius[kk])
                        W_z = W_induced[ii-1,jj,kk,2]
                        lift,phi,normal_force[kk,jj],tangential_force[kk,jj],f_s[ii,jj,kk] = \
                            UNSTEADY_SOLVER.convergence_a_aPrime(V_rel,twist, chord,\
                                                 thick_to_chord, aoa, lift_coefficient,\
                                                 drag_coefficient,separation_function,
                                                 linear_lift_coefficient,stalled_lift_coefficient,\
                                                 number_of_airfoils,\
                                                 thickness_to_chord,ii,f_s[ii-1,jj,kk],dt,time)
                    #induced wind calculation
                    F = UNSTEADY_SOLVER.tip_loss_correction(radius[kk], phi)
                    f_g,a = UNSTEADY_SOLVER.Glauert_correction(ii,W_z,V0_system4[ii,jj,:],\
                                            a_mean,local_a_calculation)
                    induced_denominator = UNSTEADY_SOLVER.denominator_induced_wind\
                                            (ii, V0_system4[ii,jj,:],f_g, W_z)
                    W_induced_quasi_steady[ii,jj,kk,:] = UNSTEADY_SOLVER.induced_wind_quasi_steady\
                                            (ii,lift,phi, radius[kk], F, induced_denominator)
                    if third_point:
                        if ii < 10:
                            tau_1,tau_2 = (1e-3,1e-3)
                        else:
                            tau_1,tau_2 = UNSTEADY_SOLVER.time_constants_induced_wind(\
                                                    a,ws_hub_height,radius[kk])
                        W_induced[ii,jj,kk,:],W_induced_intermediate[ii,jj,kk,:] = UNSTEADY_SOLVER.dynamic_wake\
                                                    (W_induced_quasi_steady[ii,jj,kk,:],\
                                                    W_induced_quasi_steady[ii-1,jj,kk,:],\
                                                    W_induced_intermediate[ii-1,jj,kk,:],\
                                                    W_induced[ii-1,jj,kk,:],
                                                    tau_1,tau_2,dt)
                    else:
                        W_induced[ii,jj,kk,:] =  W_induced_quasi_steady[ii,jj,kk,:]
        # mean value of a on the blades (calculated as the value at 0.7*R).
        if ii!=0:
            a_mean = (-W_induced[ii-1,:,8,2]/ws_hub_height).mean()
            a_values[ii] = a_mean
        
        #save the p_n, p_t, torque, power, trust array for the current time step
        final_tangential_force[ii,:,:] = tangential_force 
        final_normal_force[ii,:,:] = normal_force 
        torque[ii] = np.trapezoid(radius*tangential_force[:,0],radius)+\
                     np.trapezoid(radius*tangential_force[:,1],radius)+\
                     np.trapezoid(radius*tangential_force[:,2],radius)
        power[ii] = torque[ii]*omega;
        thrust[ii] = np.trapezoid(normal_force[:,0],radius)+\
                     np.trapezoid(normal_force[:,1],radius)+\
                     np.trapezoid(normal_force[:,2],radius)

#%% FIGURES
colors = ['#377eb8','#e41a1c']
fig = plt.figure(1)
plt.plot(time_array,power*1e-6,linestyle='--',label='$Power$',color = colors[0])
plt.plot(time_array,thrust*1e-6,linestyle='--',label='$Thrust$',color = colors[1])
plt.legend(loc="upper right",frameon= False)
plt.xlabel('$t\: [s]$')
plt.ylabel('$P\:&\:T\:[MW\:&\:MN]$')
plt.xlim([time_array[0], time_array[-1]])
plt.minorticks_on()
plt.tick_params(direction='in',right=True,top =True)
plt.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
plt.tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
plt.tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)

fig = plt.figure(2)
plt.plot(radius,final_tangential_force[-1,:,0],linestyle='--',label='$P_y$',color = colors[0])
plt.plot(radius,final_normal_force[-1,:,0],linestyle='--',label='$P_z$',color = colors[1])
plt.legend(loc="upper left",frameon= False )
plt.xlabel('$r\: [m]$')
plt.ylabel('$p_y\:&\:p_z\:[N]$')
plt.xlim([radius[0], radius[-1]])
plt.minorticks_on()
plt.tick_params(direction='in',right=True,top =True)
plt.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
plt.tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
plt.tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)


end_time = tm.perf_counter()
execution_time = end_time-start_time
print(f'The program required {execution_time:.1f} s')