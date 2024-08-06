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

def create_map(df, n_clusters):
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

print(df)

# grouped_df = df.groupby('Day')

# print(grouped_df)



# create_map(df, n_clusters)




