#%% IMPORT PACKAGES

import numpy as np

#%% INITIALIZE THE CLASS

class PitchController():
    def __init__(self,KK,Kp,Ki,w_ref,dt,max_change_pitch,\
                 pitch_max,pitch_min,I_rot,w_0=8,psi=0.7):
        self.w_ref = w_ref         # [rad/s] reference rotational speed                      
        self.KK = np.deg2rad(KK)   # [rad] gain reduction                   
        self.Kp =  Kp              # [rad/(rad/s)] gain for proportional pitch       
        self.Ki = Ki               # [rad/rad] gain for integral pitch
        self.dt = dt               # [s] time step
        self.pitch_max = np.deg2rad(pitch_max)   # [rad] max pitch angle
        self.pitch_min = np.deg2rad(pitch_min)   # [rad] min pitch angle
        self.max_change_pitch = np.deg2rad(max_change_pitch) # [rad/s] max pitch angle rate of change
        self.I_rot = I_rot         # [kg*m^2] rotor inertia
        self.w_0 = w_0             # constant of Tjaereborg rotor
        self.psi = psi             # constant of Tjaereborg rotor
    def GK_calculation(self,pitch_bef):
        '''This function determines the gain reduction at high pitch angles,
        in order to limit its variation.
        
        Parameters:
            pitch_bef (float): 
                pitch angle at the previous time step

        Returns:
            GK (float):
                gain reduction
        '''
        GK = 1/(1+pitch_bef/self.KK)
        return GK
    
    def pitch_set_calculation(self,w_bef,pitch_i_bef,pitch_bef):
        '''Determines the setpoint pitch angle from its components, the integral
        and the proportional pitch angles. It also checks the limits of the integral
        part and the setpoint value.
        
        Parameters:
            w_bef (float):
                rotational speed at the previous time step
        pitch_i_bef (float):
                integral pitch angle at the previous time step
        pitch_bef (float):
                pitch angle at the previous time step

        Returns:
            pitch_set (float):
                setpoint pitch angle
            pitch_p (float):
                current proportional pitch angle
            pitch_p (float):
                current proportional pitch angle
            GK (float):
                gain reduction
        '''
        GK = PitchController.GK_calculation(self,pitch_bef)
        pitch_p = GK*self.Kp*(w_bef-self.w_ref) # proportional pitch
        pitch_i = pitch_i_bef+GK*self.Ki*self.dt*(w_bef-self.w_ref) # integral pitch
        
        # checking value of integral pitch
        pitch_i = np.max([pitch_i,self.pitch_min])
        pitch_i = np.min([pitch_i,self.pitch_max])
        pitch_set = pitch_i+pitch_p
        
        # checking value of setpoint pitch
        pitch_set = np.max([pitch_set,self.pitch_min])
        pitch_set = np.min([pitch_set,self.pitch_max])
        
        return pitch_set,pitch_p,pitch_i,GK
    
    def inertia_pitch(self,set_bef,pitch_bef,pitch_bef2):
        '''This function takes into account the inertia in the pitch setting
        system.
        
        Parameters:
            set_bef (float):
                pitch setpoint at the previous time step
            pitch_bef (float):
                pitch angle at the previous time step
            pitch_bef2 (float):
                pitch angle at 2 previous time steps

        Returns:
        pitch (float):
            pitch angle at the current time step

        '''
        num = self.w_0**2*self.dt**2*set_bef+(2-self.w_0**2*self.dt**2)*pitch_bef+\
            (self.psi*self.w_0*self.dt-1)*pitch_bef2 # numerator of the equation
        den = 1+self.psi*self.w_0*self.dt # denominator of the equation
        pitch = num/den
        
        return pitch
    
    def checking_pitch(self,pitch_bef,pitch_now):
        '''Checking the value of the pitch angle to see if it is between the 
        limits or if it has changed too much.
        
        Parameters:
            pitch_bef (float):
                pitch angle at the previous iteration
            pitch_now (float):
                pitch angle at the current iteration

        Returns:
            pitch_now (float):
                pitch angle at the current iteration
        '''
        # max change rate with increase
        if pitch_now > pitch_bef+self.max_change_pitch*self.dt:
            pitch_now = pitch_bef+self.max_change_pitch*self.dt
        # max change rate with decrease    
        if pitch_now < pitch_bef-self.max_change_pitch*self.dt:
            pitch_now = pitch_bef-self.max_change_pitch*self.dt
        # max pitch angle
        if pitch_now >= self.pitch_max:
            pitch_now = self.pitch_max
        # min pitch angle
        if pitch_now <= self.pitch_min:
            pitch_now = self.pitch_min
            
        return pitch_now
    
    def rotational_speed(self,torque_aero,torque_gen,w_bef):
        '''Calxulates the rotational speed at the current iteration with the 
        following equation:
            I*dw/dt = M(aero)-M(torque)
        
        Parameters:
            torque_aero (float):
                mechanical torque of the rotor at the previous iteration
            torque_gen (float):
                generator torque at the previous iteration
            w_bef (float):
                rotational speed at the previous iteration

        Returns:
            w_now (float):
                rotational speed at the current iteration
        '''
        w_now = w_bef+(torque_aero-torque_gen)/self.I_rot*self.dt
        
        return w_now