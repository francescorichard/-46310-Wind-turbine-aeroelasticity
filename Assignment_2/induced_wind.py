#%% IMPORT PACKAGES

import numpy as np

#%% INITIALIZE THE CLASS

class InducedWind():
    def __init__(self,B=3,RHO=1.225,R=89.15):
        self.B = B                            # [-] Number of blades
        self.RHO = RHO                        # [kg/m^3] Density
        self.R = R                            # [m] Rotor radius
    
    def dynamic_wake(self,W_qs_now,W_qs_bef,W_int_bef,W_ind_bef,tau_1,tau_2,dt):
        '''This function calculates the induced wind for the dynamic wake 
        model from the quasi steady one and the intermediate induced wind.
        
        Parameters:
        W_qs_now (1-D array like):
            quasi steady induced wind at the current time step (now)
        W_qs_bef (1-D array like): 
            quasi steady induced wind at the previous time step (before)
        W_int_bef (1-D array like): 
            intermediate induced wind at the previous time step (before)
        W_ind_bef (1-D array like): 
            induced wind at the previous time step (before)
        tau_1 (float): 
            first time constant
        tau_2 (float): 
            second time constant
        dt (float): 
            time step of the simulation
    
        Returns:
        W_ind_now (1-D array like): 
            induced wind at the previous time step (now)
        W_int_now (1-D array like): 
            intermediate induced wind at the previous time step (now)
    
        '''
        k = 0.6 #constant for the differential equation
        H = W_qs_now + k*tau_1*(W_qs_now-W_qs_bef)/dt #right part of the differential equation
        W_int_now = H+(W_int_bef-H)*np.exp(-dt/tau_1) #intermediate induced velocity
                                                      #at the current time step  
        W_ind_now = W_int_now+(W_ind_bef-W_int_now)*np.exp(-dt/tau_2) #intermediate 
                                                                      #induced velocity
                                                                      #at the current time step 
        return W_ind_now,W_int_now
    
    def time_constants_induced_wind(self,a,V_0,r):
        '''This function determines the time constants of the differential 
        equation used to calculate the induced wind in the dynamic wake model.
        
        Parameters:
        a (foat):
            axial induction factor
        V_0 (float): 
            module of the wind speed
        r (float): 
            position on the blade
    
        Returns:
        tau_1 (float): time constant number 1
        tau_2 (float): time constant number 2
    
        '''
        a = np.min(np.array([a,0.5])) #for the model a can be maximum 0.5
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
    def Glauert_correction(iteration,W_z,vel_sys4,aa_mean,local_a_calculation):
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