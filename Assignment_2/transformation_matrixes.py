#%% IMPORT PACKAGES

import numpy as np

#%% INITIALIZE THE CLASS

class TransformationMatrixes():
    def __init__(self,theta_cone=0,theta_yaw=0,theta_pitch=0):
        self.theta_cone = theta_cone          # [rad] Cone angle
        self.theta_yaw = theta_yaw            # [rad] Yaw angle
        self.theta_pitch = theta_pitch        # [rad] Shaft pitch angle 
    
    
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
    

