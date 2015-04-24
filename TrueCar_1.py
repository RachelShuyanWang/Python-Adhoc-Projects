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
import random



###########Warning: The code will take over 20 min to run (on my laptop) #################


########################## Part I. Historical car sales transactions  ####################
''' I. Import Dataset'''
# 1.1 Read in csv data to data frame
df = pd.read_csv("hw_data_set_1.csv")
df2 = pd.read_csv("hw_data_set_1.csv")


# Leave raw dataset untouched
df1 = df
print len(df1)
print len(df1.columns)
print df1.dtypes


# 1.2 Adjust data type when necessary
# Change upper to lowercase for datetime.strptime. exp: APR - Apr
for i in xrange(len(df1)):
          df1['sales_date'][i] = df1['sales_date'][i][0:3] + df1['sales_date'][i][3:5].lower() + df1['sales_date'][i][5:9]
          df1['sales_week'][i] = df1['sales_week'][i][0:3] + df1['sales_week'][i][3:5].lower() + df1['sales_week'][i][5:9]


for i in xrange(len(df1)):
          df1['sales_date'][i] = datetime.datetime.strptime(df1['sales_date'][i], "%d%b%Y")
          df1['sales_week'][i] = datetime.datetime.strptime(df1['sales_week'][i], "%d%b%Y")

          

''' II. Outlier Data, Data Validation, Data Cleaning'''



# 2.1  Check if there is duplicate data - if so, drop it
df1_original_length = len(df1)
df1 = df1.drop_duplicates()
df1_after_length = len(df1)
# 11% of the data were duplicated
print (df1_original_length-df1_after_length)/df1_original_length

# 2.2 Check if there is missing data
print df1.isnull().sum()

# 2.3 Check the values of each variable to find outliers
print df1.describe()

# 2.3.1 Price variable
# Noticing that the max value of price is $3,000,000 - it's quite impossible
print df1[df1['price'] >= 200000]
# There are ten cars valued over $200,000, drop the outliers
df1 = df1[df1['price'] < 200000]

# 2.3.2 State variable
print df1['State'].unique()
print len(df1['State'].unique()) # 51 -- it was 50 states + DC

# 2.3.3 Date variable (sales_date, sales_week, and year)
def check_date(data):
          print max(data)
          print min(data)

check_date(df1['sales_date'])
check_date(df1['year'])

# Noticing that the max value of year is 2025 - it's impossible
print df1[df1['year'] > 2013]
# There are ten cars "sold after 2014", drop the outliers
df1 = df1[df1['year'] < 2014]

# 2.3.4 Customercash variable
print df1['customercash'].describe()
# Noticing that the max value of customercash variable is $867,300
# Drop the data when the customercash is bigger than base_msrp (which does not make sense)
df1 = df1[df1['customercash'] < df1['base_msrp']]
# 431 rows dropped, accounting for 0.3% of the whole dataset

# 2.3.5 Relatable variables (cash/finance/lease)
# There should be no cars falling in the cash and finance category at the same time
print df1[(df1['finance'] == 1) & (df1['cash'] ==1)]
print df1[(df1['lease'] == 1) & (df1['cash'] ==1)]
print df1[(df1['lease'] == 1) & (df1['finance'] ==1)]

# Other variables - They are seem fine at this moment
print df1['make'].unique()
print df1['drive_type'].unique()
print df1['door'].unique()
print df1['transmission'].unique()
print df1['bodytype'].unique()
print df1['finance'].unique()

print df1['base_msrp'].describe()
print df1['transaction_msrp'].describe()
print df1['destination'].describe()
print df1['zip'].describe()
print df1['dealercash'].describe()
print df1['customercash'].describe()
print df1['longitude'].describe()
print df1['latitude'].describe()

# The working dataset has 110049 rows, compared to raw dataset's 124028 rows, dropped 11.27% of the data
# 11% of them are duplicated, so the influence to our statistical power is very small


''' III. What factors in the dataset seem to influence prices'''

# because of colinearity of cash/finance/lease, we only use finance and cash in the model
res = ols(y=df1['price'], x=df1[['transaction_msrp','dealercash','customercash','finance','cash','destination']])
results = res.fit()

''' IV. Predicting dateset two'''
predictions = results.predict(df2[['transaction_msrp','dealercash','customercash','finance','cash','destination']])





























