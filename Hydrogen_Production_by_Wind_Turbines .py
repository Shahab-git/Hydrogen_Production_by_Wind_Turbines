import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import math
from scipy.optimize import fsolve
from scipy.interpolate import make_interp_spline
plt.rc('figure', figsize=(15,10))
df_wind_data = pd.read_csv('chart 1-1.csv', )
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

print(df_wind_data["DateTime"])
df_wind_genereation_Actual=(df_wind_data["Actual Wind Generation  (MW)"])
print(df_wind_genereation_Actual)
df_wind_genereation_Forecast=(df_wind_data['Forecast Wind Generation (MW)'])
print(df_wind_genereation_Forecast)


#plt.bar(df_wind_genereation_Forecast[minimum_velocity_Forecast],df_wind_data['Actual Wind Generation  (MW)'], color='b')
#plt.bar(df_wind_genereation[minimum_velocity_Actual],df_wind_data['Actual Wind Generation  (MW)'], color='b')
#plt.title('Minimum Velocity', fontsize=20)
#plt.ylabel('minimum_velocity', fontsize=12)
#plt.xlabel('DateTime', fontsize=12)
#plt.savefig('velocity_histogram.png', dpi=300, bbox_inches='tight')
