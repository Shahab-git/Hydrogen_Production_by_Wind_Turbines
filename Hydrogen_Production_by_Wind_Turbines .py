import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import fsolve
from scipy.interpolate import make_interp_spline
plt.rc('figure', figsize=(15,10))
df_wind_data = pd.read_csv('chart 1-1.csv', )
df_wind_data.columns
df_wind_data.head()
df_wind_data.info()

minimum_velocity_Actual = df_wind_data["Actual Wind Generation  (MW)"].min()
maximum_velocity_Actual = df_wind_data["Actual Wind Generation  (MW)"].max()
minimum_velocity_Forecast = df_wind_data["Forecast Wind Generation (MW)"].min()
maximum_velocity_Forecast = df_wind_data["Forecast Wind Generation (MW)"].max()
print('Minimum Velocity Actual = '+str(minimum_velocity_Actual)+'MW')
print('Maximum Velocity Actual = '+str(maximum_velocity_Actual)+'MW')
print('Minimum Velocity Forecast = '+str(minimum_velocity_Forecast)+'MW')
print('Maximum Velocity Forecast = '+str(maximum_velocity_Forecast)+'MW')

#print(df_wind_data["DateTime"])
df_wind_genereation_Actual=(df_wind_data["Actual Wind Generation  (MW)"])
#print(df_wind_genereation_Actual)
df_wind_genereation_Forecast=(df_wind_data['Forecast Wind Generation (MW)'])
#print(df_wind_genereation_Forecast)

#Calculating Voltage reversible & Overpotential for PEM Electrolyzer
from numpy import log as ln
NH2O = 1.5
F = 96485.3365
Istack = NH2O*2*F
print("Istack: ",Istack)
Ncell = 1000                                                                        #Number of cell per stack
Nstack = 100                                                                        #Number of stack needed
Icell = Istack/Ncell/Nstack                                                         #Current per cell
T0 = 273                                                                            #Zero Degree Celsius to Kelvin (K)
Tcell = T0+80                                                                       #Cell Operating Temperature
Vrev = 1.5298-1.5421*1e-03*Tcell+9.523*1e-05*Tcell*ln(Tcell)+9.84*10e-08*Tcell**2   #Voltage reversible calculation
print("Vrev: ",Vrev)
Vact = 0.0514*Icell+0.2098                                                          #Calculating activation overpotential
print("Vact: ",Vact)
Vohm = 0.08*Icell                                                                   #Calculating ohmic overpotential
print("Vohm: ",Vohm)
Vovr = Vrev+Vact+Vohm                                                               #Overall voltage
print("Vovr: ",Vovr)
Watt=Istack*Vovr
print("Watt: ",Watt)

#Production Hydrogen
LHV_H2 = 119950                             #kJ/kg
Ncell = 1000                                #Number of cell per stack
delta_H=286                                 #Hydrogen higher heating value(Joule)
z=1                                         #Number of electrons exchanged during the reaction
v_cell= 1.229                               #Cell voltage

#Q_heat_H2O"
enthalpy= 285.9                             #kJ/mol
J = 6000
E = 76000
F = 96485.3365                              #Faraday's constant (C/mol)
T_PEME_K = 353
T_0 = 273
P_0=101.3                                   #[kPa]
Q_H2O=5.061
HHV_H2=141860                               #(kJ/kg)
Q_cell=0
N_dot_H2O_reacted=J / (2 * F)               #Molar rate of H2O consumed in the reaction
N_dot_H2_out = N_dot_H2O_reacted            #Molar outlet flow rate of H2
η=(N_dot_H2_out*HHV_H2)/(E+Q_cell+Q_H2O)
P=df_wind_genereation_Actual                #available energy
Vc=1.229                                    #Cell voltage
F = 96485.3365                              #Faraday's constant (C/mol)
M=1.00794                                   #molar mass of hydrogen (g/mol)
m_H2=(P/(Vc*2*F))*((M*η))
print("m_H2: ","\n",m_H2)

