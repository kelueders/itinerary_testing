import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import haversine_distances
from geopy.distance import geodesic
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from collections import defaultdict

from trip_391 import trip_391
from trip_385 import trip_385

# function to convert the trip object into the format required to turn into a dataframe
def extract_id_lat_long(trip):

    id_lat_long = {
        "id": [],
        "lat": [],
        "long": []
    }
    
    places = trip["places"]
    # pprint(places)

    for place in places.values():

        id_ = place['id']
        id_lat_long['id'].append(id_)

        lat = place['lat']
        id_lat_long['lat'].append(lat)

        long = place['long']
        id_lat_long['long'].append(long)

    return id_lat_long

def extract_trip_duration(trip):
    duration = len(trip["days"].keys())
    return duration


data = extract_id_lat_long(trip_385)
df = pd.DataFrame(data)
n_days = extract_trip_duration(trip_385)

# Convert lat/long to radians for haversine calculation
df['lat_rad'] = np.radians(df['lat'])
df['long_rad'] = np.radians(df['long'])

# Function to calculate distance between two points in kilometers
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance (in kilometers) between two points 
    on the earth (specified in decimal degrees).
    """
    R = 6371  # Radius of the Earth in kilometers
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) * np.sin(dlon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

# Calculate pairwise haversine distances for all locations
distances = haversine_distances(df[['lat_rad', 'long_rad']].values)

# Function to group locations based on proximity and limit per day
def group_locations(distances, max_locations_per_day):
    """
    Groups locations based on proximity while limiting the number of locations per day.
    """
    n_locations = len(distances)
    assignments = [0] * n_locations
    current_day = 1
    current_group = [0]
    locations_in_current_day = 1

    for i in range(1, n_locations):
        # Find the closest location to the current group
        closest_index = np.argmin(distances[current_group, i])

        # If closest location is within a certain distance threshold (e.g., 50km)
        # and we haven't reached the limit for the current day, assign it to the current day
        if distances[current_group[closest_index], i] <= 50 and locations_in_current_day < max_locations_per_day:
            current_group.append(i)
            assignments[i] = current_day
            locations_in_current_day += 1
        else:
            # Move to the next day
            current_day += 1
            assignments[i] = current_day
            current_group = [i]
            locations_in_current_day = 1

    return assignments

# Group locations with a maximum of 4 per day
assignments = group_locations(distances, 4)
df['Day'] = assignments

# Print the grouped DataFrame
print(df)





# Create the map
fig = plt.figure(figsize=(12, 8))
m = Basemap(projection='merc', llcrnrlat=20, urcrnrlat=50, llcrnrlon=-130, urcrnrlon=-60, resolution='l')
m.drawcoastlines()
m.fillcontinents(color='lightgray', lake_color='aqua')
m.drawmapboundary(fill_color='aqua')

# Get the coordinates of the locations
lons = df['long'].values
lats = df['lat'].values

# Plot the locations with different colors for each cluster
colors = ['red', 'blue', 'green', 'purple']  # Adjust colors as needed
for i in range(n_days):
    cluster_indices = df[df['Day'] == i].index
    x, y = m(lons[cluster_indices], lats[cluster_indices])
    m.plot(x, y, 'o', markersize=8, color=colors[i], label=f'Day {i+1}')

# Add a legend
plt.legend()

# Set the title
plt.title('KMeans Clustering of Locations')

# Show the plot
plt.show()