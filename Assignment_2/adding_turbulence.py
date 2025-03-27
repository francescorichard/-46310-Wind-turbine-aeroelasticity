import numpy as np
from scipy import signal
from scipy import interpolate

#%% INITIALIZE THE CLASS
class AddingTurbulence():
    def __init__(self,dt,time_steps,n2=32,n3=32,Ly=180,Lz=180):
        #self.umean = umean
        self.deltat = dt
        #self.deltax = self.deltat*self.umean
        self.n1 = 2048
        self.n2 = n2
        self.n3 = n3
        #self.Lx = int(self.deltax*(self.n1-1))
        self.Ly = Ly
        self.Lz = Lz
        self.deltay=self.Ly/(self.n2-1)
        self.deltaz=self.Lz/(self.n3-1)
        # self.deltat=deltax/self.umean
        self.x_turb = np.arange(0,self.n3)*self.deltaz+(119-(self.n3-1)*self.deltaz/2)
        self.y_turb = np.arange(0,self.n2)*self.deltay-((self.n2-1)*self.deltay)/2
        #self.z_turb = np.arange(0,self.n1)*self.deltax
    
    def smoothing_spectra(self,f,raw_spectra,n_decade=15):
        '''This function smoothes the signal of the raw spectrum.

        Parameters:
            f : 
                frequency array
            raw_spectra : raw spectrum

        Returns:
            f_smoothed:
                frequency of the smoothed spectrum
            S_smoothed:
                smoothed spectrum

        '''
        # Compute min and max frequencies (ignoring zero)
        f_min = np.min(f[f > 0])  # Minimum frequency (ignoring 0)
        f_max = np.max(f)  # Maximum frequency
        
        # Compute logarithmic bins
        log_f_min = np.log10(f_min)
        log_f_max = np.log10(f_max)
        n_bins = round(n_decade * (log_f_max - log_f_min))  # Total number of bins
        
        # Define frequency bins (logarithmic spacing)
        log_bins = np.logspace(log_f_min, log_f_max, n_bins)
        
        # Initialize smoothed spectra
        S_smoothed = np.zeros(n_bins - 1)
        f_smoothed = np.zeros(n_bins - 1)
        
        # Apply smoothing filter
        for i in range(n_bins - 1):
            # Find the indices of frequencies in the current bin
            bin_indices = np.where((f >= log_bins[i]) & (f < log_bins[i + 1]))[0]
        
            if bin_indices.size > 0:
                # Mean of the frequency in the bin
                f_smoothed[i] = np.mean(f[bin_indices])
                
                # Mean of the spectra in the bin
                S_smoothed[i] = np.mean(raw_spectra[bin_indices])
        
        # Remove zero entries if some bins are empty
        f_smoothed = f_smoothed[f_smoothed > 0]
        S_smoothed = S_smoothed[S_smoothed > 0]
        return f_smoothed,S_smoothed

    def load(self,filename, N=(32, 32)):
        """Load mann turbulence box

        Parameters:
            filename : 
                Filename of turbulence box
            N : 
                Number of grid points

        Returns:
            turbulence_box : nd_array

        """
        data = np.fromfile(filename, np.dtype('<f'), -1)
        if len(N) == 2:
            ny, nz = N
            nx = len(data) / (ny * nz)
            assert nx == int(nx), "Size of turbulence box (%d) does not match ny x nz (%d), nx=%.2f" % (
                len(data), ny * nz, nx)
            nx = int(nx)
        else:
            nx, ny, nz = N
            assert len(data) == nx * ny * \
                nz, "Size of turbulence box (%d) does not match nx x ny x nz (%d)" % (len(data), nx * ny * nz)
        return data.reshape(nx, ny * nz)
    
    def calculating_turbulence_field(self,filename):
        '''This function reshape the turbulence field from 2d to 3d
        
        Parameters:
            filename:
                name of the turbulence data\'s file
        Return:
            uplane:
                reshaped turbulence field
            x_turb:
                coordinates in the x direction (vertical)
            y_turb:
                coordinates in the y direction (normal)
        '''
        u=AddingTurbulence.load(self,filename,  N=(self.n1, self.n2, self.n3))
        uplane= np.reshape(u, (self.n1, self.n2, self.n3))
        return uplane,self.x_turb,self.y_turb
    
    def calculating_psd(self,input_data,pwelch):
        '''This function determines the power spectra density of a signal in the 
        frequency domain. The latter can be done with either the welch function or
        an implemented one.
        
        Parameters:
            input_data:
                data of which you want to determine the spectrum
            pwelch:
                boolean value. If True, the spectrum is determined with pwelch, 
                otherwhise it uses the other one
        
        Returns:
            f:
                frequency of the spectrum
            S:
                spectrum of the signal
        '''
        time=np.arange(self.deltat, self.n1*self.deltat+self.deltat, self.deltat)
        n = len(input_data)
        fs=1/(time[1]-time[0])
        if pwelch:
            f, S = signal.welch(input_data, fs, nperseg=1024)
        else:
            fn = 1/(2*self.deltat)
            U = np.fft.fft(input_data)
            A = U[:n//2 + 1]
            S = (1/(fs*n))*np.abs(A)**2
            S[1:-1] *= 2
            f = fs*np.arange(0,n//2+1)/n
            f,S = AddingTurbulence.smoothing_spectra(self,f,S)
        return f,S
        
    def interpolating_turbulence(self,time,x_point,y_point,velocity):
        '''This function interpolates the point of an element in the turbulence field
        to determine the turbulence component in that point.
            
            Parameters:
    
                time : 
                    time iteration .
                x_point : 
                    x coordinate of the point.
                y_point : 
                    y coordinate of the point.
                velocity : 
                    turbulence field at that time iteration.
    
            Returns:
                interp_func:
                    value of the turbulence component in the wanted point
    
        '''    
        interp_func = interpolate.interpn((self.y_turb, self.x_turb),velocity[time,...],\
                                          np.array([y_point,x_point]),method='linear')
        return float(interp_func)
        