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
minimum_velocity = df_wind_data["Actual Wind Generation  (MW)"].min()
maximum_velocity = df_wind_data["Actual Wind Generation  (MW)"].max()
print('Minimum Velocity = '+str(minimum_velocity)+'MW')
print('Maximum Velocity = '+str(maximum_velocity)+'MW')
median_velocity = np.arange(0.5, math.ceil(minimum_velocity))
lower_limit_velocity = np.floor(median_velocity)
upper_limit_velocity = np.ceil(median_velocity)
dictionary_frequency_distribution = {'Lower Limit Velocity [m/s]':lower_limit_velocity,'Upper Limit Velocity [m/s]':upper_limit_velocity, 'Median Velocity [m/s]':median_velocity, 'Absolute Frequency':np.zeros}
df_frequency_distribution = pd.DataFrame(dictionary_frequency_distribution)
df_frequency_distribution.head(15)
for i in df_wind_data['DateTime']:
    for j in range(0, df_frequency_distribution.shape[0]):
        if (i < df_frequency_distribution['Upper Limit Velocity [m/s]'][j]) and (i > df_frequency_distribution['Lower Limit Velocity [m/s]'][j]):
            df_frequency_distribution['Absolute Frequency'][j] = 1 + df_frequency_distribution['Absolute Frequency'][j]
df_frequency_distribution.insert(loc = (df_frequency_distribution.shape[1]),column = 'Relative Frequency', value = df_frequency_distribution['Absolute Frequency']/df_frequency_distribution['Absolute Frequency'].sum())
df_frequency_distribution.insert(loc = (df_frequency_distribution.shape[1]),column = 'Relative Frequency [%]', value = 100*df_frequency_distribution['Relative Frequency'])
df_frequency_distribution.head(15)
df_wind_genereation=(df_wind_data["Actual Wind Generation  (MW)"],df_wind_data['Forecast Wind Generation (MW)'])
plt.bar(df_wind_genereation[minimum_velocity],df_wind_data['Actual Wind Generation  (MW)'], color='b')
plt.title('Minimum Velocity', fontsize=20)
plt.ylabel('Actual Wind Generation  (MW)', fontsize=12)
plt.xlabel('minimum_velocity', fontsize=12)
plt.savefig('velocity_histogram.png', dpi=300, bbox_inches='tight')
