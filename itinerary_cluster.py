import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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

    # print(id_lat_long)

# extract_id_lat_long(trip_391)

# Sample data (replace with your own data)

# data = {
#     'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Philadelphia', 
#              'Phoenix', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'],
#     'lat': [40.7128, 34.0522, 41.8781, 29.7604, 39.9526, 33.4484, 29.4241, 32.7157, 32.7767, 37.3382],
#     'long': [-74.0060, -118.2437, -87.6298, -95.3698, -75.1652, -112.0740, -98.4936, -117.1611, -96.7970, -121.8863]
# }
data = extract_id_lat_long(trip_385)
df = pd.DataFrame(data)
n_clusters = extract_trip_duration(trip_385)

print(f"Number of days: {n_clusters}")

# Select features for clustering
X = df[['lat', 'long']]

# Feature scaling (important for KMeans)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create the KMeans model
kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # Try different values of n_clusters

# Fit the model to the data
kmeans.fit(X_scaled)

# Get the cluster labels for each data point
df['Day'] = kmeans.labels_

# 5. Group locations by cluster and sort by proximity
cluster_groups = defaultdict(list)
for i, row in df.iterrows():
    cluster_groups[row['Day']].append(row)

# for cluster in cluster_groups:
#     cluster_groups[cluster] = sorted(cluster_groups[cluster], key=lambda row: row['Day'])

print("CLUSTER GROUPS")
pprint(cluster_groups)

# # 6. Distribute locations evenly across days
# selected_locations = []
# remaining_locations = []
# for i in range(len(df)):
#     if len(cluster_groups[i % n_clusters]) > 0 and len(selected_locations) % 4 < 4:  # Ensure 4 location limit
#         selected_locations.append(cluster_groups[i % n_clusters].pop(0))
#     else:
#         remaining_locations.extend(cluster_groups[i % n_clusters])

# # 7. Assign remaining locations to the day with the fewest locations
# cluster_counts = [len(cluster_groups[i]) for i in range(n_clusters)]
# min_cluster_index = cluster_counts.index(min(cluster_counts))

# for location in remaining_locations:
#     if len(cluster_groups[min_cluster_index]) < 4:  # Ensure 4 location limit
#         cluster_groups[min_cluster_index].append(location)
#         cluster_counts[min_cluster_index] += 1

# # Create a new DataFrame from selected locations
# df_sorted = pd.DataFrame(selected_locations)
# for i in range(n_clusters):
#     for location in cluster_groups[i]:
#         df_sorted = df_sorted._append(cluster_groups[i])

# # Create a new DataFrame from selected locations
# # df_sorted = pd.DataFrame(selected_locations)

# # # Print or visualize the results
# print(f"Sorted DataFrame:/n{df_sorted}")

# # 8. Print the sorted locations per day
# for day in range(n_clusters):
#     print(f"Day {day+1}:")
#     print(df_sorted[df_sorted['Day'] == day]['id'])
#     print("---")






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
for i in range(n_clusters):
    cluster_indices = df[df['Day'] == i].index
    x, y = m(lons[cluster_indices], lats[cluster_indices])
    m.plot(x, y, 'o', markersize=8, color=colors[i], label=f'Day {i+1}')

# Add a legend
plt.legend()

# Set the title
plt.title('KMeans Clustering of Locations')

# Show the plot
plt.show()



