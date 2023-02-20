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
#id;year;day;datetm;min;ws_25;wd_25;tp_25;ws_50;wd_50;tp_50
#DateTime,Actual Wind Generation  (MW),Forecast Wind Generation (MW)
minimum_velocity_Actual = df_wind_data["Actual Wind Generation  (MW)"].min()
maximum_velocity_Actual = df_wind_data["Actual Wind Generation  (MW)"].max()
minimum_velocity_Forecast = df_wind_data["Forecast Wind Generation (MW)"].min()
maximum_velocity_Forecast = df_wind_data["Forecast Wind Generation (MW)"].max()
print('Minimum Velocity Actual = '+str(minimum_velocity_Actual)+'MW')
print('Maximum Velocity Actual = '+str(maximum_velocity_Actual)+'MW')
print('Minimum Velocity Forecast = '+str(minimum_velocity_Forecast)+'MW')
print('Maximum Velocity Forecast = '+str(maximum_velocity_Forecast)+'MW')

median_velocity_Actual = np.arange(0.5, math.ceil(minimum_velocity_Actual))
median_velocity_Forecast = np.arange(0.5, math.ceil(minimum_velocity_Forecast))
lower_limit_velocity_Actual = np.floor(median_velocity_Actual)
upper_limit_velocity_Actual = np.ceil(median_velocity_Actual)
lower_limit_velocity_Forecast = np.floor(median_velocity_Forecast)
upper_limit_velocity_Forecast = np.ceil(median_velocity_Forecast)
dictionary_frequency_distribution_Actual = {'Lower Limit Velocity Actual [m/s]':lower_limit_velocity_Actual,'Upper Limit Velocity Actual [m/s]':upper_limit_velocity_Actual, 'Median Velocity Actual [m/s]':median_velocity_Actual, 'Absolute Frequency Actual':np.zeros}
dictionary_frequency_distribution_Forecast = {'Lower Limit Velocity Forecast [m/s]':lower_limit_velocity_Forecast,'Upper Limit Velocity Forecast [m/s]':upper_limit_velocity_Forecast, 'Median Velocity Forecast [m/s]':median_velocity_Forecast, 'Absolute Frequency Forecast ':np.zeros}
df_frequency_distribution_Actual = pd.DataFrame(dictionary_frequency_distribution_Actual)
df_frequency_distribution_Forecast = pd.DataFrame(dictionary_frequency_distribution_Forecast)
df_frequency_distribution_Actual.head(15)
df_frequency_distribution_Forecast.head(15)

#for i in df_wind_data['Actual Wind Generation  (MW)']:
    #for j in range(0, df_frequency_distribution_Actual.shape[0]):
         #if (i < df_frequency_distribution_Actual['upper_limit_velocity_Actual [m/s]'][j]) and (i > df_frequency_distribution_Actual['Lower Limit Velocity_Actual [m/s]'][j]):
            #df_frequency_distribution_Actual['Absolute Frequency'][j] = 1 + df_frequency_distribution_Actual['Absolute Frequency'][j]
#df_frequency_distribution_Actual.insert(loc = (df_frequency_distribution_Actual.shape[1]),column = 'Relative Frequency Actual', value = df_frequency_distribution_Actual['Absolute Frequency Actual']/df_frequency_distribution_Actual['Absolute Frequency Actual'].sum())
#df_frequency_distribution_Actual.insert(loc = (df_frequency_distribution_Actual.shape[1]),column = 'Relative Frequency Actual [%]', value = 100*df_frequency_distribution_Actual['Relative Frequency Actual'])
df_wind_genereation_Actual=(df_wind_data["Actual Wind Generation  (MW)"],df_wind_data['Forecast Wind Generation (MW)'])
print(df_wind_genereation_Actual)
#plt.bar(df_wind_genereation_Actual[minimum_velocity_Actual],df_wind_data['Actual Wind Generation  (MW)'], color='b')
#plt.show()

#for i in df_wind_data['Forecast Wind Generation (MW)']:
    #for j in range(0, df_frequency_distribution_Forecast.shape[0]):
         #if (i < df_frequency_distribution_Forecast['upper limit velocity Forecast [m/s]'][j]) and (i > df_frequency_distribution_Forecast['Lower Limit Velocity Forecast [m/s]'][j]):
            #df_frequency_distribution_Forecast['Absolute Frequency Forecast'][j] = 1 + df_frequency_distribution_Forecast['Absolute Frequency Forecast'][j]
#df_frequency_distribution_Forecast.insert(loc = (df_frequency_distribution_Forecast.shape[1]),column = 'Relative Frequency Forecast', value = df_frequency_distribution_Forecast['Absolute Frequency Forecast']/df_frequency_distribution_Forecast['Absolute Frequency Forecast'].sum())
#df_frequency_distribution_Forecast.insert(loc = (df_frequency_distribution_Forecast.shape[1]),column = 'Relative Frequency Forecast [%]', value = 100*df_frequency_distribution_Forecast['Relative Frequency Forecast'])
#df_wind_genereation_Forecast=(df_wind_data["Forecast Wind Generation  (MW)"],df_wind_data['Forecast Wind Generation (MW)'])
#print(df_wind_genereation_Forecast)
#plt.bar(df_wind_genereation_Forecast[minimum_velocity_Forecast],df_wind_data['Forecast Wind Generation  (MW)'], color='b')
#plt.show()

#plt.bar(df_wind_genereation_Forecast[minimum_velocity_Forecast],df_wind_data['Actual Wind Generation  (MW)'], color='b')
#plt.bar(df_wind_genereation[minimum_velocity_Actual],df_wind_data['Actual Wind Generation  (MW)'], color='b')
#plt.title('Minimum Velocity', fontsize=20)
#plt.ylabel('minimum_velocity', fontsize=12)
#plt.xlabel('DateTime', fontsize=12)
#plt.savefig('velocity_histogram.png', dpi=300, bbox_inches='tight')
