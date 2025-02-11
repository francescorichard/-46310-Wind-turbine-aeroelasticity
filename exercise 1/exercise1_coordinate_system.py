
#%% IMPORT PACKAGES

import numpy as np
import matplotlib.pyplot as plt

#%% INITIALIZE VALUES

omega = 0.62 #rotational speed [rad/s]
theta_cone = np.deg2rad(0)
theta_yaw = np.deg2rad(0)
theta_pitch = np.deg2rad(0)
#theta_blade = np.deg2rad(0)
H =119 #hub height
L_s = 7.1 #shaft length [m]
R = 89.15 # [m]
dt = 0.15 #delta t [s]
x_b = 70 #[m]
TOTAL_TIME = 15 # [s]
shear_exponent = 0
ws_hub_height = 10 # [m/s]
a_x = 3.32 # [m] it should change with height
#%% DEFINE THE TRANSFORMATION MATRIXES
'''
Calculate all the transformation matrixes to pass from the ground (easy system to
measure wind speed) to the blade (where we calculate the velocity), and reverse.                                                                  
'''
# matrix between system 1 (ground) and system 2 (nacelle)
def matrix_a1(theta_yaw):
     a_1 = np.array([[1, 0, 0],
                     [0, np.cos(theta_yaw), np.sin(theta_yaw)],
                     [0, -np.sin(theta_yaw), np.cos(theta_yaw)]])
     return a_1
 
def matrix_a2(theta_pitch):
     a_2 = np.array([[np.cos(theta_pitch), 0, -np.sin(theta_pitch)],
                     [0, 1, 0],
                     [np.sin(theta_pitch), 0, np.cos(theta_pitch)]])
     return a_2
 
def matrix_a12(a_1,a_2):
    a_3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    a_12 = np.matmul(np.matmul(a_1,a_2),a_3)
    a_21 = a_12.transpose()
    return a_12,a_21

def matrix_a23(theta_blade):
    a_23 = np.array([[np.cos(theta_blade), np.sin(theta_blade), 0],
                    [-np.sin(theta_blade), np.cos(theta_blade), 0],
                    [0, 0, 1]])
    return a_23
 
def matrix_a34(theta_cone):
    a_34 = np.array([[np.cos(theta_cone), 0, -np.sin(theta_cone)],
                    [0, 1, 0],
                    [np.sin(theta_cone), 0, np.cos(theta_cone)]])
    return a_34

def matrix_a14(a_34,a_23,a_12):
    a_14 = np.matmul(a_34,np.matmul(a_23,a_12))
    a_41 = a_14.transpose()
    return a_14,a_41

#%% DETERMINING THE POSITION OF A POINT ON THE BALDE IN SYSTEM 1
'''the final position is the sum of the radial positions
   of the three systems: ground, nacelle, and blade.
   
   r_groud = (H, 0, 0)
   r_nacelle = a_21*(0, 0, -L_s)
   r_blade = a_41*(x_blade, 0, 0)
   
   r_tot = r_ground + r_nacelle + r_blade
'''
def position_point_system1(H,L_s,x_b,a_21,a_41):
    r_ground = np.array([[H],[0], [0]])
    r_nacelle = np.matmul(a_21,np.array([[0],[0], [-L_s]]))
    r_blade = np.matmul(a_41,np.array([[x_b],[0], [0]]))
    
    r_final = r_ground + r_nacelle + r_blade
    return r_final.flatten()

#%% DEFINE POSITION OF BALDES IN TIME

def position_blade(omega,t):
    '''
    This function determines the azimuth position of the blades at time t.
    '''
    theta_blade_1 = omega*t
    theta_blade_2 = theta_blade_1+2/3*np.pi
    theta_blade_3 = theta_blade_1+4/3*np.pi
    return theta_blade_1,theta_blade_2,theta_blade_3


#%% FINAL CALCULATION FOR MATRIXES AND POSITION POINT
def final_calculation_position(theta_yaw,theta_blade,theta_cone,theta_pitch,H, L_s, x_b):
     '''
     This function determines the position of the point on the blade from the 
     system 1 (ground). With zero angle of yaw, tilt, and cone the
     '''
     a_1 = matrix_a1(theta_yaw)
     a_2 = matrix_a2(theta_pitch)
     a_12, a_21 = matrix_a12(a_1,a_2)
     a_23 = matrix_a23(theta_blade)
     a_34 = matrix_a34(theta_cone)
     a_14, a_41 = matrix_a14(a_34,a_23,a_12)
     r_point = position_point_system1(H, L_s, x_b, a_21, a_41)
     return r_point,a_1,a_2,a_12,a_23,a_34,a_14,a_21,a_41
 
#%% CLCULATE VELOCITY AT THE POINT
'''
If the blade passes in front of the tower, the tower itsleft will have an effect on
the wind.
'''
def velocity_system1(x,H,ws_hub_height,shear_exponent,min_height,a_x):
    v0_x = ws_hub_height*(x[0]/H)**shear_exponent
    if x[0] != min_height:
        velocity_sys1 = np.array([0, 0, v0_x])
    else:
        r = np.sqrt(x[1]**2+x[2]**2)
        v_r = x[2]/r*v0_x*(1-(a_x/r)**2)
        v_theta = x[1]/r*v0_x*(1+(a_x/r)**2)
        v_y= x[1]/r*v_r-x[2]/r*v_theta
        v_z = x[2]/r*v_r+x[1]/r*v_theta
        velocity_sys1 = np.array([0, v_y, v_z])
    return velocity_sys1

#%% TRANSFER THE VELOCITY FROM SYSTEM 1 TO SYSTEM 4

def velocity_system4(a_14,vel_sys1):
    vel_sys4 = np.matmul(a_14,vel_sys1)
    return vel_sys4
#%% MAIN
time_steps = int(TOTAL_TIME/dt)
position_blade1 = np.zeros((time_steps,3))
position_blade2 = np.zeros((time_steps,3))
position_blade3 = np.zeros((time_steps,3))
V0_system1 = np.zeros((time_steps,3))
V0_system4 = np.zeros((time_steps,3))

azimuthal_angle_blade1 = np.zeros((time_steps,1))
time_array = np.linspace(0, TOTAL_TIME, time_steps)
for ii in range(0,time_steps):
    time = ii*dt
    theta_blade_1,theta_blade_2,theta_blade_3 = position_blade(omega,time)
    azimuthal_angle_blade1[ii] = theta_blade_1
    for jj in range(0,3):
        if jj == 0:
            theta_blade = theta_blade_1
            position_blade1[ii,:],a_1,a_2,a_12,a_23,a_34,a_14,a_21,a_41 = final_calculation_position(theta_yaw, theta_blade, theta_cone, theta_pitch, H, L_s, x_b)
            V0_system1[ii,:] = velocity_system1(position_blade1[ii,:], H, ws_hub_height, shear_exponent,np.min(position_blade1[ii,0]),a_x)
            V0_system4[ii,:] = velocity_system4(a_14,V0_system1[ii,:])
        elif jj == 1:
            theta_blade = theta_blade_2
            position_blade2[ii,:],a_1,a_2,a_12,a_23,a_34,a_14,a_21,a_41 = final_calculation_position(theta_yaw, theta_blade, theta_cone, theta_pitch, H, L_s, x_b)

        else:
            theta_blade = theta_blade_3
            position_blade3[ii,:],a_1,a_2,a_12,a_23,a_34,a_14,a_21,a_41 = final_calculation_position(theta_yaw, theta_blade, theta_cone, theta_pitch, H, L_s, x_b)


#colors for plots thaken from ColorBrewer
colors = ['#1b9e77','#d95f02','#7570b3']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(time_array,position_blade1[:,0],position_blade1[:,1],marker='o',ms=4,mfc='w',linestyle='--',label='$x\:position\: blade\:1$',color=colors[0],linewidth=2)
ax.plot(time_array,position_blade2[:,0],position_blade2[:,1],marker='o',ms=4,mfc='w',linestyle='--',label='$x\: position\: blade\: 2$',color=colors[1],linewidth=2)
ax.plot(time_array,position_blade3[:,0],position_blade3[:,1],marker='o',ms=4,mfc='w',linestyle='--',label='$x\: position\: blade\: 3$',color=colors[2],linewidth=2)
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
ax.set_xlabel('t [s]')
ax.set_ylabel('x position [m]')
ax.set_zlabel('y position [m]')
ax.set_xlim([0, time_array[-1]])
ax.minorticks_on()
ax.tick_params(direction='in',right=True,top =True)
ax.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
ax.tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
ax.tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)
#yticks = np.arange(0,16.1,4)
# plt.yticks(yticks)

fig = plt.figure(2)
plt.plot(np.rad2deg(azimuthal_angle_blade1),V0_system4[:,2],linestyle='--',label='$V_z\:blade\:1$',color=colors[0],linewidth=2)
plt.plot(np.rad2deg(azimuthal_angle_blade1),V0_system4[:,1],linestyle='--',label='$V_y\: blade\:1$',color=colors[1],linewidth=2)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.xlabel('$\:theta\: [^\circ]$')
plt.ylabel('velocity [m/s]')
plt.xlim([5, np.rad2deg(azimuthal_angle_blade1)[-1]])
plt.minorticks_on()
plt.tick_params(direction='in',right=True,top =True)
plt.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
plt.tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
plt.tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)
#yticks = np.arange(0,16.1,4)
#plt.yticks(yticks)
