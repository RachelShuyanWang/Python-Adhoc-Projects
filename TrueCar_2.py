import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from pandas.stats.api import ols

import matplotlib.pyplot as plt
from pandas import DataFrame
from pylab import *
from scipy.cluster.vq import *
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.cluster import KMeans

''' I. Import Dataset'''
# 1.1 Import dataset from csv to dataframe
df_raw = pd.read_csv("trims.csv")

# 1.2 Copy the dateset, leave original dataset untouched
df = df_raw

''' II. Outlier Data, Data Validation, Data Cleaning'''
# 2.1  Check if there is duplicate data - if so, drop it
len_original= len(df)
df = df.drop_duplicates()
len_drop_dup = len(df)
# No duplicated data
print len_original - len_drop_dup

# 2.2 Check if there is missing data
print df.isnull().sum()
# Drop the one row that has missing data
df = df[df.isnull() == False]

# 2.3 Check outliers from min and max value
print df.describe()

# Noticing there is negative value for pct_discount - it does not make sense
# (or does it? - when the demand for a certain car is extremely high - which is not true judging from the transaction data)
print df[df['pct_discount']<0]
print len(df[df['pct_discount']<0])

# Drop the 20 rows of negative pct_discount
# (20 is very small compared to 1829, so it has minor affet to our statistical power)
df = df[df['pct_discount']>0]

''' III. Descriptive data'''
# 3.1 Unique values for each variable
print df['make'].unique(), len(df['make'].unique())
print df['tc_body'].unique(), len(df['tc_body'].unique())

# 3.2 Descriptive data for cross tabulation/ stats
# 3.2.1 Categorical data - make, tc_body, model
print df['make'].value_counts()
print df['tc_body'].value_counts()
print df['model'].value_counts()

#  Plot them!
# Plot the stats of car make
make_stat = DataFrame(df['make'].value_counts())
make_stat.plot(kind = "bar", legend=False)
plt.title("Stats of Car Make")
plt.xlabel("Car Make")
plt.ylabel("Number of Car IDs")
plt.show()

# Plot the stats of car body
tc_body_stat = DataFrame(df['tc_body'].value_counts())
tc_body_stat.plot(kind = "bar", legend=False)
plt.title("Stats of Car Body")
plt.xlabel("Car Body")
plt.ylabel("Number of Car IDs")
plt.show()

# 3.2.2 Cross tabulation of car make and body
print pd.crosstab(df.make, df.tc_body)
tc_body_make_stat = DataFrame(pd.crosstab(df.make, df.tc_body))
# Stacked bar plot of make and body

# We want pretty plot
# These are the "Tableau 20" colors as RGB.  
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  
  
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.  
for i in range(len(tableau20)):  
    r, g, b = tableau20[i]  
    tableau20[i] = (r / 255., g / 255., b / 255.)  

plt.gca().grid(True)
tc_body_make_stat.plot(kind='bar', stacked=True, color = tableau20)
plt.title("Stats of Car Body and Make")
plt.xlabel("Car Body")
plt.ylabel("Number of Car IDs")
#  the default plot color and style are PRETTY UGLY, I would definitely improve given more time
plt.show()

# 3.2.3 Add a column - the real price of vehicle
# real price = msrp * (1 - pct_discount)
df['real_price'] = df['msrp'] * (1 - df['pct_discount'])

''' IV. Method One. K-means Clustering'''
# Assumption 1. Visitors only want to see recommendation of same tc_body
# Assumption 2. Visitors are willing to see recommendation of other make and model
# Assumption 3. Visitors have a fixed budget, and would like to see recommendations in the same price range
# Assumption 4. Visitors would like to accept cars of the same taste/"popularity" - denoted by number of national sales 
# Assumption 5. Visitors would like to see as many recommendations as possible, thus the recommendation system should always try to recommend 5 vehicles

### Note that this k-means clustering include an visualization of the clustering process

# 4.1  A function that takes a sub-dataset that has only one tc_body type as input
# and output the labels of trim_ids from clustering package
def cluster_by_body(data):
          # Plot raw data
          plt.scatter(data['real_price'], data['n_ind_transactions'])
          plt.show()
          
          # Apply kmeans from sklearn - get centroids and labels
          X = np.column_stack((np.array(data['real_price']), np.array(data['n_ind_transactions'])))
          kmeans = KMeans(n_clusters = int((len(data)/5) * 0.5)) # 0.8 is a parameter we can change to control the number of clusters
          kmeans.fit(X)

          centroids = kmeans.cluster_centers_
          labels = kmeans.labels_
          print(centroids)
          print(labels)

          colors = ["g.","r.","y.","b","m","c"] * 100 # make sure it's long enough so it doesn't overflow

          # Plot labeled data with different colors
          for i in range(len(X)):
                    # print ("coordinate:", X[i], "label:", labels[i])
                    plt.plot(X[i][0],  X[i][1],  colors[labels[i]], markersize = 10)

          # Plot the centroids of each cluster as "x"
          plt.scatter(centroids[:, 0], centroids[:, 1], marker = "x", s = 50, linewidths = 3, zorder = 10)
          plt.show()
          return labels

# 4.2 A function that takes an int id as an input and out put upto 5 similar 
def recommend_five_vehicles(id):
          if id not in df['trim_id'].tolist():
                    return "This id doesn't exist in the dataset, try again."
          else:
                    body_type = df[df['trim_id'] == id]['tc_body']
                    subdf = df[df['tc_body'] == body_type.values[0]]
                    subdf['label'] = (cluster_by_body(subdf)).tolist()
                    subdf_label = subdf[subdf['trim_id'] == id]['label']
                    if len(subdf[(subdf['label'] == subdf_label.values[0]) & (subdf['trim_id'] != id)]) <=5:
                              return subdf[(subdf['label'] == subdf_label.values[0]) & (subdf['trim_id'] != id)]['trim_id']
                    else:
                              return random.sample((subdf[(subdf['label'] == subdf_label.values[0]) & (subdf['trim_id'] != id)]['trim_id']), 5)

# 4.3 Test of the method
print recommend_five_vehicles(df['trim_id'][1])

# 4.4 Reflection
# Pro:1.Can be scaled 2. No need for prior info
# Con: 1.Random sampling 2. Have to decide number of clusters

''' V. Method Two. Hybrid Recommendation System'''

# This Hybrid Recommendation System provides a mixed group of recommendations with different features
# 

# Assumption 1. Visitors only want to see recommendation of same tc_body
# Assumption 2. Vehicles of the same make are considered on the same branch
# Assumption 3. Vehicles with in 20% +/- price are considered to be in the same price range
# Assumption 4. Recommendation 1: best seller car + same branch + same price range - if the orginal searched vehicle is the best seller, recommend the second best
# Assumption 5. Recommendation 2: same branch + 10% more expensive - just offer a better option
# Assumption 6. Recommendation 3: same branch + same price range + the largest discount
# Assumption 7. Recommendation 4: best seller car + another branch + same price range
# Assumption 8. Recommendation 5: a similar car + another branch + same price range + same popularity range

# 5.1 Analysis
# Pro: 1.Probably more efficient 2.Easy to control/adjust with large sized dataset, 3.Reproducable process
# Con: 1.Hard to scale, 2. Needs some prior information

