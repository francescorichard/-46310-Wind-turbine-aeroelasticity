#%% IMPORT PACKAGES

import numpy as np
#%% INITIALIZE THE CLASS

class PositionDefinition():
    def __init__(self,TRANSFORMATION_MATR,H=119,L_s=7.1):
        self.H = H                            # [m] Hub height
        self.L_s = L_s                        # [m] Nacelle length
        self.TRANSFORMATION_MATR = TRANSFORMATION_MATR


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
        r_nacelle = self.TRANSFORMATION_MATR.a_21 @ np.array([[0],[0], [-self.L_s]])
        r_blade = self.TRANSFORMATION_MATR.a_41 @ np.array([[x_b],[0], [0]])
        r_final = r_ground + r_nacelle + r_blade
        return r_final.flatten()
    
    def position_blade(self,omega,theta_adding,dt):
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
        theta_blade_1 = theta_adding+omega*dt
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
         self.TRANSFORMATION_MATR.a_1 = self.TRANSFORMATION_MATR.matrix_a1()
         self.TRANSFORMATION_MATR.a_2 = self.TRANSFORMATION_MATR.matrix_a2()
         self.TRANSFORMATION_MATR.a_12, self.TRANSFORMATION_MATR.a_21 =self.TRANSFORMATION_MATR.matrix_a12()
         self.TRANSFORMATION_MATR.a_23 = self.TRANSFORMATION_MATR.matrix_a23(theta_blade)
         self.TRANSFORMATION_MATR.a_34 = self.TRANSFORMATION_MATR.matrix_a34()
         self.TRANSFORMATION_MATR.a_14, self.TRANSFORMATION_MATR.a_41 = self.TRANSFORMATION_MATR.matrix_a14()
         self.TRANSFORMATION_MATR.r_point = self.position_point_system1(x_b)
         return self.TRANSFORMATION_MATR.r_point
