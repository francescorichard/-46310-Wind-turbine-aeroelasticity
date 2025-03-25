#%% IMPORT PACKAGES

import numpy as np

#%% INITIALIZE THE CLASS

class SavingData():
    def __init__(self,number_of_airfoils):
        self.number_of_airfoils = number_of_airfoils

    
    def opening_files(self):
        self.cylinder = np.loadtxt('C:\\COPENAGHEN PRIMO ANNO\\AEROELASTICITY\\turbine_data\\cylinder_ds.txt')
        self.FFA_W3_301 = np.loadtxt('C:\\COPENAGHEN PRIMO ANNO\\AEROELASTICITY\\turbine_data\\FFA-W3-301_ds.txt');
        self.FFA_W3_360 = np.loadtxt('C:\\COPENAGHEN PRIMO ANNO\\AEROELASTICITY\\turbine_data\\FFA-W3-360_ds.txt');
        self.FFA_W3_480 = np.loadtxt('C:\\COPENAGHEN PRIMO ANNO\\AEROELASTICITY\\turbine_data\\FFA-W3-480_ds.txt');
        self.FFA_W3_600 = np.loadtxt('C:\\COPENAGHEN PRIMO ANNO\\AEROELASTICITY\\turbine_data\\FFA-W3-600_ds.txt');
        self.FFA_W3_2411 = np.loadtxt('C:\\COPENAGHEN PRIMO ANNO\\AEROELASTICITY\\turbine_data\\FFA-W3-241_ds.txt');
        self.blade_data = np.loadtxt('C:\\COPENAGHEN PRIMO ANNO\\AEROELASTICITY\\turbine_data\\bladedat.txt');
    
    def storing_data(self):
        radius = self.blade_data[:,0]
        
        # save data in different matrix.I want to save a matrix with every variable.
        # Angle of attack, lift coefficient, drag coefficient, and thrust coefficient
        total_data = np.concatenate((self.cylinder,
                                    self.FFA_W3_600,
                                    self.FFA_W3_480,
                                    self.FFA_W3_360,
                                    self.FFA_W3_301,
                                    self.FFA_W3_2411),axis=1)
        
        #initialize the lift, drag, angle of attack, and thickness to chord
        #ratio matrixes. I also initialize the separation function, the linear lift
        #coefficient and the stalled lift coefficient, needed for the dynamic stall.
        aoa = np.zeros((total_data.shape[0],self.number_of_airfoils)) #angle of attack
        lift_coefficient = np.zeros((total_data.shape[0],self.number_of_airfoils))
        drag_coefficient = np.zeros((total_data.shape[0],self.number_of_airfoils))
        separation_function = np.zeros((total_data.shape[0],self.number_of_airfoils))
        linear_lift_coefficient = np.zeros((total_data.shape[0],self.number_of_airfoils))
        stalled_lift_coefficient = np.zeros((total_data.shape[0],self.number_of_airfoils))
        position = 0
        for kk in range(0,self.number_of_airfoils):
            aoa[:,kk] = total_data[:,position]
            lift_coefficient[:,kk] = total_data[:,position+1]
            drag_coefficient[:,kk] = total_data[:,position+2]
            separation_function[:,kk] = total_data[:,position+4]
            linear_lift_coefficient[:,kk] = total_data[:,position+5]
            stalled_lift_coefficient[:,kk] = total_data[:,position+6]
            position += 7
        return aoa,lift_coefficient,drag_coefficient,separation_function,linear_lift_coefficient,\
               stalled_lift_coefficient,radius
    


