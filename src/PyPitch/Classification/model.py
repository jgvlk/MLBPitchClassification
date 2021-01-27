from matplotlib import pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from pprint import pprint
import statsapi

from PyPitch.classification.utils import import_pitcher_data
from PyPitch.classification.viz import show_release_point, show_pitch_location


### [1] EDA & TRANSFORMATIONS ###
#################################

# IMPORT DATASET #
player = statsapi.lookup_player('Lester', season=2019)
statsapi.lookup_player(620443, season=2019)
pprint(player)

player_id = player[0]['id']
player_name = player[0]['firstLastName']

ret, df = import_pitcher_data('Lester', season=2020)

df['Pitcher'].head()


# VIZ: RELEASE POINT BY PARK #
########
# TO DO:
#   - Query sample from each park
########
df_pitch = pd.DataFrame(None)
df_park = pd.DataFrame(None)

df_pitch['x_release'] = df['x0']
df_pitch['y_release'] = df['y0']
df_pitch['z_release'] = df['z0']
df_pitch['x_location'] = df['px']
df_pitch['z_location'] = df['pz']
df_pitch['ParkID'] = df['ParkID']

df_park['ColorParkID'] = df['ParkID']
df_park = df_park.drop_duplicates()

colors = iter(cm.rainbow(np.linspace(0, 1, len(df_park))))
df_park['color'] = None
for index,row in df_park.iterrows():
    df_park['color'][index] = next(colors)

df_pitch = df_pitch.merge(df_park, left_on='ParkID', right_on='ColorParkID')


fig_release_point = plt.figure()
ax = Axes3D(fig_release_point)

x = df_pitch['x_release']
y = df_pitch['y_release']
z = df_pitch['z_release']
park_id = df_pitch['ParkID']

for x,y,z,color in zip(x.iteritems(), y.iteritems(), z.iteritems(), park_id.iteritems()):
    ax.scatter(x[1], y[1], z[1], s=10, c=color[1])

plt.show()

