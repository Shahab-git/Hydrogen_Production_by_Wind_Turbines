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
df_wind_genereation=(df_wind_data["Actual Wind Generation  (MW)"],df_wind_data['Forecast Wind Generation (MW)'])
plt.bar(df_wind_genereation[minimum_velocity],df_wind_data['Actual Wind Generation  (MW)'], color='b')
plt.title('Minimum Velocity', fontsize=20)
plt.ylabel('Actual Wind Generation  (MW)', fontsize=12)
plt.xlabel('minimum_velocity', fontsize=12)
plt.savefig('velocity_histogram.png', dpi=300, bbox_inches='tight')
