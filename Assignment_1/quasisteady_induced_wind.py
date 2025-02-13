#%% IMPORT PACKAGES

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib as mpl
#%% Opening files
'''We need the file for the various airfoils and the data 
   of the blade. The actual airfoils of the blade will be
   determined with interpolation between the given airfoils.
'''
cylinder = np.loadtxt('C:\\COPENAGHEN PRIMO ANNO\\AEROELASTICITY\\turbine_data\\cylinder_ds.txt')
FFA_W3_301 = np.loadtxt('C:\\COPENAGHEN PRIMO ANNO\\AEROELASTICITY\\turbine_data\\FFA-W3-301_ds.txt');
FFA_W3_360 = np.loadtxt('C:\\COPENAGHEN PRIMO ANNO\\AEROELASTICITY\\turbine_data\\FFA-W3-360_ds.txt');
FFA_W3_480 = np.loadtxt('C:\\COPENAGHEN PRIMO ANNO\\AEROELASTICITY\\turbine_data\\FFA-W3-480_ds.txt');
FFA_W3_600 = np.loadtxt('C:\\COPENAGHEN PRIMO ANNO\\AEROELASTICITY\\turbine_data\\FFA-W3-600_ds.txt');
FFA_W3_2411 = np.loadtxt('C:\\COPENAGHEN PRIMO ANNO\\AEROELASTICITY\\turbine_data\\FFA-W3-241_ds.txt');
blade_data = np.loadtxt('C:\\COPENAGHEN PRIMO ANNO\\AEROELASTICITY\\turbine_data\\bladedat.txt');
number_of_airfoils = 6;

#%% save data in different matrix
'''I want to save a matrix with every variable. Angle of attack,
   lift coefficient, drag coefficient, and thrust coefficient
'''
total_data = np.concatenate((cylinder,
                            FFA_W3_600,
                            FFA_W3_480,
                            FFA_W3_360,
                            FFA_W3_301,
                            FFA_W3_2411),axis=1)
#I initialize the matrixes
thickness_to_chord = np.array((100, 60, 48, 36, 30.1, 24.1))
aoa = np.zeros((total_data.shape[0],number_of_airfoils))
lift_coefficient = np.zeros((total_data.shape[0],number_of_airfoils))
drag_coefficient = np.zeros((total_data.shape[0],number_of_airfoils))

position = 0
for kk in range(0,number_of_airfoils):
    aoa[:,kk] = total_data[:,position]
    lift_coefficient[:,kk] = total_data[:,position+1]
    drag_coefficient[:,kk] = total_data[:,position+2]
    position += 7

radius = blade_data[:,0]

#%% INITIALIZE VALUES

B = 3 # number of blades
RATED_POWER = 10 # [MW]
V_IN = 4 # cut in speed [m/s]
V_OUT = 25 #cut-out speed [m/s]
RHO = 1.225 # density [kg/m^3]
TIP_PITCH = np.deg2rad(0) # tip pitch chosen arbitrarly [m/s]
omega = 0.72 #rotational speed [rad/s]
theta_cone = np.deg2rad(0)
theta_yaw = np.deg2rad(0)
theta_pitch = np.deg2rad(0)
#theta_blade = np.deg2rad(0)
H =119 #hub height
L_s = 7.1 #shaft length [m]
R = 89.15 # [m]
dt = 0.15 #delta t [s]
TOTAL_TIME = 15 # [s]
shear_exponent = 0
ws_hub_height = 8 # [m/s]
a_x = 0 # [m] it should change with height

#%% BEM CODE
def convergence_a_aPrime(V_rel,tip_pitch,twist,c,rho,
                         thick_to_chord,aoa,lift_coefficient,
                         drag_coefficient,number_of_airfoils,thickness_to_chord):

    abs_V_rel = np.sqrt(V_rel[1]**2+V_rel[2]**2)
    phi = np.arctan2(V_rel[2],-V_rel[1])
    local_pitch = twist+tip_pitch
    angle_attack = phi-local_pitch
    
    # interpolating
    clthick = np.zeros((number_of_airfoils,1))
    cdthick = np.zeros((number_of_airfoils,1))
    for kk in range(0,number_of_airfoils):
        clthick[kk] = interpolate.interp1d(aoa[:,kk], lift_coefficient[:,kk], kind='linear')(np.rad2deg(angle_attack))
        cdthick[kk] = interpolate.interp1d(aoa[:,kk], drag_coefficient[:,kk], kind='linear')(np.rad2deg(angle_attack))
    clift = interpolate.interp1d(thickness_to_chord,clthick[:,0], kind='linear')(thick_to_chord)
    cdrag = interpolate.interp1d(thickness_to_chord,cdthick[:,0], kind='linear')(thick_to_chord)
    lift = 1/2*rho*c*clift*abs_V_rel**2
    drag = 1/2*rho*c*cdrag*abs_V_rel**2
    infinitesimal_tang_force = lift*np.sin(phi)-drag*np.cos(phi)
    infinitesimal_norm_force = lift*np.cos(phi)+drag*np.sin(phi)
        
    return lift,phi,infinitesimal_norm_force,infinitesimal_tang_force
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
    a_12 = np.matmul(np.matmul(a_3,a_2),a_1)
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
 
#%% CALCULATE VELOCITY AT THE POINT
'''
If the blade passes in front of the tower, the tower itsleft will have an effect on
the wind.
'''
def velocity_system1(x,H,ws_hub_height,shear_exponent,a_x):
    v0_x = ws_hub_height*(x[0]/H)**shear_exponent
    if x[0] > H:
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

#%% CALCULATE INDUCED WIND

def induced_wind(iteration,B,lift,phi,rho,radial_pos,F,denominator):
    '''
    Parameters
    ----------
    iteration : time step.
    B : number of blades
    lift : lift per chord length.
    phi : flow angle.
    rho : density.
    radial_pos : radial position on the blade of the considered element.
    F : Prandtl's tip loss factor.
    denominator : |V_0+f_g*W_n|.

    Returns
    -------
    W_y : induced wind y-component.
    W_z : induced wind z-component.
    '''
    if iteration == 0:
        W_z = 0
        W_y = 0
    else:
        W_z = -B*lift*np.cos(phi)/(4*np.pi*rho*radial_pos*F*denominator)
        W_y = -B*lift*np.sin(phi)/(4*np.pi*rho*radial_pos*F*denominator)
    return np.array([0, W_y, W_z])

def denominator_induced_wind(iteration,vel_sys4,f_g,W_z):
    if iteration == 0:
        W_z = 0
    denominator = np.sqrt(vel_sys4[1]**2+(vel_sys4[2]+f_g*W_z)**2)
    return denominator

def Glauert_correction(iteration,W_z,vel_sys4,aa):
    if iteration == 0:
        W_z = 0
    V_0 = np.sqrt(vel_sys4[1]**2+vel_sys4[2]**2)
    aa = -W_z/V_0
    if aa <= 1/3:
        f_g = 1
    else:
        f_g = 1/4*(5-3*aa)
    return f_g  

def tip_loss_correction(B,R,r,phi):
    ff = np.divide(B*(R-r),2*r*np.sin(np.abs(phi)))
    F = 2/np.pi*np.arccos(np.exp(-ff))
    return F
#%% DETERMINE RELATIVE VELOCITY

def relative_velocity(vel_sys4,induced_wind,omega,radial_position,theta_cone):
    '''
    Parameters
    ----------
    vel_sys4 : wind speed in blade system.
    induced_wind : induced wind in blade system.
    omega : rotational speed.
    radial_position : radial position  of the element.
    theta_cone : cone angle.

    Returns
    -------
    V_rel = relative velocity at the radial position.

    '''
    V_rel_y = vel_sys4[1]+induced_wind[1]-omega*radial_position*np.cos(theta_cone)
    V_rel_z = vel_sys4[2]+induced_wind[2]
    return np.array([0, V_rel_y, V_rel_z])

#%% MAIN
#vector inizialization
time_steps = int(TOTAL_TIME/dt)
position_blades = np.zeros((time_steps,B,3))
V0_system1 = np.zeros((time_steps,B,3))
V0_system4 = np.zeros((time_steps,B,3)) 
W_induced = np.zeros((time_steps,B,radius.shape[0],3))
azimuthal_angle_blade1 = np.zeros((time_steps,1))
time_array = np.linspace(0, TOTAL_TIME, time_steps)
theta_blade = np.zeros((B,1))
final_tangential_force = np.zeros((time_steps,radius.shape[0],B))
final_normal_force = np.zeros((time_steps,radius.shape[0],B))
torque = np.zeros((time_steps,1))
power = np.zeros((time_steps,1))
thrust = np.zeros((time_steps,1))
a_values = np.zeros((time_steps,1))
a_mean = 0

for ii in range(0,time_steps):
    tangential_force = np.zeros((radius.shape[0],B))
    normal_force = np.zeros((radius.shape[0],B))
    time = ii*dt #time update
    theta_blade[0],theta_blade[1],theta_blade[2] = position_blade(omega,time) #saving the position of the blades at current time
    azimuthal_angle_blade1[ii] = theta_blade[0]
    for jj in range(0,B):
            theta_blade_considered = float(theta_blade[jj])
            for kk in range(0,radius.shape[0]-1):
                #calculates the velocity in system 1 and 4
                position_blades[ii,jj,:],a_1,a_2,a_12,a_23,a_34,a_14,a_21,a_41 = final_calculation_position(theta_yaw, theta_blade_considered, theta_cone, theta_pitch, H, L_s, radius[kk])
                V0_system1[ii,jj,:] = velocity_system1(position_blades[ii,jj,:], H, ws_hub_height, shear_exponent,a_x)
                V0_system4[ii,jj,:] = velocity_system4(a_14,V0_system1[ii,jj,:])
                twist = np.deg2rad(blade_data[kk,1])
                chord = blade_data[kk,2]
                thick_to_chord = blade_data[kk,3]
                if ii == 0:
                    V_rel = relative_velocity(V0_system4[ii,jj,:], [0, 0, 0], omega, radius[kk], theta_cone)
                    W_z = 0   
                else:
                    V_rel = relative_velocity(V0_system4[ii,jj,:], W_induced[ii-1,jj,kk,:], omega, radius[kk], theta_cone)
                    W_z = W_induced[ii-1,jj,kk,2]
                lift,phi,normal_force[kk,jj],tangential_force[kk,jj] = convergence_a_aPrime(V_rel, TIP_PITCH, twist, chord, RHO, thick_to_chord, aoa, lift_coefficient, drag_coefficient, number_of_airfoils, thickness_to_chord)
                F = tip_loss_correction(B, R, radius[kk], phi)
                induced_denominator = denominator_induced_wind(ii, V0_system4[ii,jj,:], Glauert_correction(ii,W_z,V0_system4[ii,jj,:],a_mean), W_z)
                W_induced[ii,jj,kk,:] = induced_wind(ii, B, lift, phi, RHO, radius[kk], F, induced_denominator)
    if ii!=0:
        a_mean = (-W_induced[ii-1,:,8,2]/ws_hub_height).mean()
        a_values[ii] = a_mean
    final_tangential_force[ii,:,:] = tangential_force 
    final_normal_force[ii,:,:] = normal_force 
    torque[ii] = np.trapezoid(radius*tangential_force[:,0],radius)+np.trapezoid(radius*tangential_force[:,1],radius)+np.trapezoid(radius*tangential_force[:,2],radius)
    power[ii] = torque[ii]*omega;
    thrust[ii] = np.trapezoid(normal_force[:,0],radius)+np.trapezoid(normal_force[:,1],radius)+np.trapezoid(normal_force[:,2],radius)

#%% plot commands

#size
mpl.rcParams['figure.figsize'] = (12,8)

#font sizeof label, title, and legend
mpl.rcParams['font.size'] = 14
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['legend.fontsize'] = 14

#%% FIGURES
colors = ['#1b9e77','#d95f02']
fig = plt.figure(1)
plt.plot(time_array,power*1e-6,linestyle='--',label='$Power$',linewidth=2,color = colors[0])
plt.plot(time_array,thrust*1e-6,linestyle='--',label='$Thrust$',linewidth=2,color = colors[1])
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.xlabel('$t\: [s]$')
plt.ylabel('$P\:&\:T\:[MW\:&\:MN]$')
plt.xlim([time_array[0], time_array[-1]])
plt.minorticks_on()
plt.tick_params(direction='in',right=True,top =True)
plt.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
plt.tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
plt.tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)

fig = plt.figure(2)
plt.plot(radius,final_tangential_force[-1,:,0],linestyle='--',label='$P_y$',linewidth=2)
plt.plot(radius,final_normal_force[-1,:,0],linestyle='--',label='$P_z$',linewidth=2)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.xlabel('$r\: [m]$')
plt.ylabel('$p_y\:&\:p_z\:[N]$')
plt.xlim([radius[0], radius[-1]])
plt.minorticks_on()
plt.tick_params(direction='in',right=True,top =True)
plt.tick_params(labelbottom=True,labeltop=False,labelleft=True,labelright=False)
plt.tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
plt.tick_params(direction='in',which='major',length=10,bottom=True,top=True,right=True,left=True)