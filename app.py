# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 16:57:39 2021

@author: antwi
"""

#Pandas and numpy for data manipulation
import pandas as pd
import numpy as np
np.random.seed(8)
import streamlit as st

# Matplotlib and seaborn for plotting 
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['figure.figsize'] = (10,5)
plt.style.use('ggplot')

import seaborn as sns

from sklearn.cluster import KMeans

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Finding Optimal Location")
st.markdown("Using linear Programming and KMeans Clustering to find the optimal location")
st.subheader("By Yaw Antwi")

st.title("Using KMeans Clustering")

st.write("The data")

uploaded_files = st.file_uploader("Upload CSV", type="csv", accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        file.seek(0)
    uploaded_data_read = [pd.read_csv(file,encoding='latin-1') for file in uploaded_files]
    df = pd.concat(uploaded_data_read)

st.write(df.head())
#df.head()
st.write("Shape of Data")
df.shape
st.write("Column Names")
df.columns


st.write("Description of Dataset")
st.write(df.describe())

st.write("Plotting original coordinates of Dataset")
st.write(sns.scatterplot(data=df, x = 'Address.Latitude' , y = 'Address.Longitude'))
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

import plotly.graph_objects as go

df['text'] = df['Address.FormattedAddress'] +'Phone Number:'+ df['PhoneNumber'].astype(str)

fig = go.Figure(data=go.Scattergeo(
        locationmode = 'USA-states',
        lon = df['Address.Longitude'],
        lat = df['Address.Latitude'],
        text = df['text'],
        mode = 'markers',
        marker = dict(
            opacity = 0.8,
            reversescale=True,
            autocolorscale = False,
            symbol = 'circle',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale = 'Blues',
            cmin = 0,
            color = df['ID'],
            cmax = df['ID'].max(),
        )))

fig.update_layout(
        title = 'Target location in USA<br>',
        geo = dict(
            scope='usa',
            projection_type='albers usa',
            showland = True
        ),
    )

st.plotly_chart(fig)

from streamlit_folium import folium_static
import folium

st.write("Map of original Coordinates")
m= folium.Map(location=[df['Address.Latitude'].mean(),df['Address.Longitude'].mean()],zoom_start=4,
             tiles='openstreetmap')
for _, row in df.iterrows():
    folium.CircleMarker(
    location=[row['Address.Latitude'],row['Address.Longitude']],
    radius=5,
    popup = row['Address.FormattedAddress'],
    color='#1787FE',
    fill=True,
    fill_color='#1787FE').add_to(m)
folium_static(m)

# need to drop the Address.name column
df1 = df[['Address.Latitude','Address.Longitude','ID']]

# # determining the k using the elbow method 
# #from the graph k = 6
SSE = []

for i in range (1,24):
     kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10,random_state = 0)
     kmeans.fit(df1)
     SSE.append(kmeans.inertia_)

fig3 = plt.figure()    
plt.plot(range(1,24),SSE) 
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('sse')
st.write("Elbow Method")
st.plotly_chart(fig3)

from scipy.spatial.distance import cdist
import random

def k_means(X, K):
#Keep track of history so you can see k-means in action
    centroids_history = []
    labels_history = []
    rand_index = np.random.choice(X.shape[0], K)
    centroids = X[rand_index]
    centroids_history.append(centroids)
    while True:
# Euclidean distances are calculated for each point relative to
# centroids, #and then np.argmin returns
# the index location of the minimal distance - which cluster a point
# is #assigned to
        labels = np.argmin(cdist(X, centroids), axis=1)
        labels_history.append(labels)
#Take mean of points within clusters to find new centroids:
        new_centroids = np.array([X[labels == i].mean(axis=0)
                                  for i in range(K)])
        centroids_history.append(new_centroids)
        
# If old centroids and new centroids no longer change, k-means is
# complete and end. Otherwise continue
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return centroids, labels, centroids_history, labels_history


df1 = df1.values
centroids, labels, centroids_history, labels_history = k_means(df1, 6)
st.text("Displaying Centroid Values")
centroids
st.text("Displaying Labels")
labels
st.text("Dataset without Clusters Added")
df1 = pd.DataFrame(df1,columns =['Address.Latitude','Address.Longitude','ID'])
st.write(df1.head())

st.text("Data with Cluster Column added")
df1['cluster']= labels
st.write(df1.head())

st.subheader("Map of Clustering")
sns.scatterplot(data = df1, x = 'Address.Latitude', y = 'Address.Longitude', hue = 'cluster',style = "cluster", palette="deep", s = 100)
st.pyplot()

k = 6
centroids
lats = [centroids[i][0] for i in range(k)]
df1['clat'] = df1['cluster'].map(lambda x: lats[x])
longs = [centroids[i][1] for i in range(k)]
df1['clong'] = df1['cluster'].map(lambda x: longs[x])
df1.head()

st.write("Centroid Coordinates Scatter plot")
sns.scatterplot(data=df1, x = 'clat' , y = 'clong')
st.pyplot()

st.text("Facility Location Clusters")
sns.scatterplot(df1['Address.Latitude'],df1['Address.Longitude'], c =df1['cluster'], s = 100)
plt.scatter(df1['clat'],df1['clong'], marker ='*', s = 100, c= 'red')
st.pyplot()

from sklearn.metrics import silhouette_score

st.write(f'Silhouette Score: {silhouette_score(df1, labels)}')

st.write("Map of Centroids of Locations")
m1= folium.Map(location=[df1['clat'].mean(),df1['clong'].mean()],zoom_start=4,
             tiles='openstreetmap')
for _, row in df1.iterrows():
    folium.Marker(
    location=[row['clat'],row['clong']],
    popup = row['cluster'],
    color='#1787FE',
    fill=True,
    fill_color='#1787FE').add_to(m1)
folium_static(m1)

import geopandas as gpd
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import tqdm
from tqdm.notebook import tqdm

df1['geom'] = df1['clat'].map(str) + ',' + df1['clong'].map(str)

locator = Nominatim(user_agent='myGeocoder', timeout=10)
rgeocode = RateLimiter(locator.reverse, min_delay_seconds=0.001)

tqdm.pandas()

df1['address'] = df1['geom'].progress_apply(rgeocode)
st.write(df1.head())

df1.drop('geom', inplace=True,axis=1)

st.write(df1.head())


st.write("Map of centriods of location and map of original coordinates")
m2= folium.Map(location=[df['Address.Latitude'].mean(),df['Address.Longitude'].mean()],zoom_start=4,
             tiles='openstreetmap')
tooltip = 'Click me!'
for _, row in df1.iterrows():
    folium.Marker(
    location=[row['clat'],row['clong']],
    popup = row['address'],
    tooltip=tooltip, icon= folium.Icon(color= 'red',icon='info-sign')
    ).add_to(m2)
for _, row in df.iterrows():   
    folium.Marker(
    location=[row['Address.Latitude'],row['Address.Longitude']],
    popup =row['Address.FormattedAddress'],
    tooltip=tooltip
    ).add_to(m2)
folium_static(m2)



st.title("Using Linear Programming to find the next location")
# Annalyzing the linear Programming Approach
#Pandas and numpy for data manipulation


# need to drop the Address.name column
df4 = df[['Address.Latitude','Address.Longitude']]
def format_equation(M,p):
    C = [0 for _ in range(2 * p + 2 * len(M) + len(M) * p + 2 * p * len(M))]
    start = 2 * p + 2 * len(M)
    end = start + len(M) * p
    for i in range(start, end):
        C[i] = 1
    return C

def format_left_in(M,p):
    A = [[0 for i in range(len(M) * (2 + 3 * p) + p * 2)] for _ in range(len(M) * 5 * p)]
    x = 0
    y = 1
    d = p * 2 + len(M) * 2
    a = p * 2
    b = p * 2 + 1
    dx = p * 2 + len(M) * (2 + p)
    dy = p * 2 + len(M) * (2 + p) + 1
    it = 0
    for i in range(0, len(M)*p):
        # constraint 1
        A[it][dx] = 1
        A[it][dy] = 1
        A[it][d] = -1
        # constraint 2
        A[it + 1][x] = 1
        A[it + 1][a] = -1
        A[it + 1][dx] = -1
        # constraint 3
        A[it + 2][x] = -1
        A[it + 2][a] = 1
        A[it + 2][dx] = -1
        # constraint 4
        A[it + 3][y] = 1
        A[it + 3][b] = -1
        A[it + 3][dy] = -1
        # constraint 5
        A[it + 4][y] = -1
        A[it + 4][b] = 1
        A[it + 4][dy] = -1
        # update indexes
        # consider the next (xi,yi) when the last element is reached
        if i % len(M) == len(M) - 1:
            x += 2
            y += 2
            d = p * 2 + len(M) * 2
            a = p * 2
            b = p * 2 + 1
        else :
            a += 2
            b += 2
            d += 1
        dx += 2
        dy += 2
        it += 5
    return A

def format_right_in(M,p):
    b = [0 for _ in range(len(M) * 5 * p)]
    return b

def format_left_eq(M,p):
    a_eq = [[0 for _ in range(p * 2 + len(M) * (2 + 3 * p))] for _ in range(len(M) * 2)]
    nb_one = p * 2
    for i in range(len(M) * 2):
        a_eq[i][i+nb_one] = 1
    return a_eq

def format_right_eq(M,p):
    b_eq = []
    for a, b in M: 
        b_eq.append(a)
        b_eq.append(b)
    return b_eq

def compute_simplex(M,p):
    C    = format_equation(M, p)
    a_in = format_left_in(M, p)
    b_in = format_right_in(M, p)
    a_eq = format_left_eq(M, p)
    b_eq = format_right_eq(M,p)
    res = linprog(C, A_ub=a_in, b_ub=b_in, A_eq=a_eq, b_eq=b_eq)
    # adjust 0f in nf for n decimals precision
    return ((res.x[0]),(res.x[1]))

def cluster(M,p):
    data = np.vstack(M)
    means, _ = kmeans(data, p)
    cluster_indexes, _ = vq(data, means)
    clusters = [[] for _ in range(p)]
    for i in range(len(cluster_indexes)):
        index = cluster_indexes[i]
        clusters[index].append((data[i][0], data[i][1]))
    return clusters

import scipy
from typing import List
from typing import Tuple
from scipy.cluster.vq import vq,kmeans,whiten
from scipy.optimize import linprog

st.title("Implementing A Linear Programming Approach")
st.subheader("The Original Data")
st.write(df.head())

df2 = df[['Address.Latitude','Address.Longitude']]
df2['Address.Longitude']= abs(df2[['Address.Longitude']])
st.text("Conversion to absolute values")
st.write(df2.head())

M = df2.values
def solve(M,p):
    clusters = cluster(M, p)
    results = []
    for c in clusters:
        results.append(compute_simplex(c, 1))
    return results
p=15
sol = solve(M,p)
sol
b = pd.DataFrame(sol, columns= ['lat','long'])
st.text("Obtain absolute values of coordinates for computation")
b['long']= -abs(b['long'])
st.write(b.head())

st.text("Optimal Facility Location")
sns.scatterplot(data=df, x = 'Address.Latitude' , y = 'Address.Longitude')
plt.scatter(data = b, x='lat',y = 'long')
st.pyplot()


import geopandas as gpd
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import tqdm
from tqdm.notebook import tqdm

b['geom'] = b['lat'].map(str) + ',' + b['long'].map(str)

locator = Nominatim(user_agent='myGeocoder', timeout=10)
rgeocode = RateLimiter(locator.reverse, min_delay_seconds=0.001)

tqdm.pandas()

b['address'] = b['geom'].progress_apply(rgeocode)
st.write(b.head())

b.drop('geom', inplace=True,axis=1)

st.write(b.head())


st.write("Map of potential Locations")
m4= folium.Map(location=[df1['clat'].mean(),df1['clong'].mean()],zoom_start=4,
             tiles='openstreetmap')
for _, row in b.iterrows():
    folium.Marker(
    location=[row['lat'],row['long']],
    popup = row['address'],
    tooltip = tooltip).add_to(m4)
folium_static(m4)


st.write("Map of potential locations and map of original coordinates")
m5= folium.Map(location=[df['Address.Latitude'].mean(),df['Address.Longitude'].mean()],zoom_start=4,
             tiles='openstreetmap')
tooltip = 'Click me!'
for _, row in b.iterrows():
    folium.Marker(
    location=[row['lat'],row['long']],
    popup = row['address'],
    tooltip=tooltip, icon= folium.Icon(color= 'red',icon='info-sign')
    ).add_to(m5)
for _, row in df.iterrows():   
    folium.Marker(
    location=[row['Address.Latitude'],row['Address.Longitude']],
    popup =row['Address.FormattedAddress'],
    tooltip=tooltip
    ).add_to(m5)
folium_static(m5)
