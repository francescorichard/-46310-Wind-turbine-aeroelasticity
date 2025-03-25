#%% IMPORT PACKAGES

import numpy as np

#%% INITIALIZE THE CLASS

class UndisturbedWindSpeed():
    def __init__(self,TRANSFORMATION_MATR,H=119,shear_exponent=0):

        self.H = H                            # [m] Hub height
        self.shear_exponent = shear_exponent  # [-] Velocity shear exponent
        self.TRANSFORMATION_MATR = TRANSFORMATION_MATR

    def velocity_system1(self,x,ws_hub_height,gain,offset,tower_shadow):
        '''This function determines the velocity of the point on the blade 
        from system 1 (ground).
        
        Parameter:
            x (float): 
                position of the point ins system 1
            ws_hub_height (float):
                wind speed at hub height
            height_model:
                linear regression of the radius of the tower
            
        Return:
            vel_sys1 (1-D array like): 
                velocity of the point in system 1 (ground)
        '''
        if tower_shadow and x[0]<=self.H:
            a_x = x[0]*gain+offset
        else:
            a_x = 0
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
        vel_sys4 = self.TRANSFORMATION_MATR.a_14 @ vel_sys1
        return vel_sys4