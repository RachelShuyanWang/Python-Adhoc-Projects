# Set up
import csv
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


''' Part I. Load, clean and transform data'''
# Read in data from two csv
contacts = pd.read_csv('takehome_contacts.csv')
assign = pd.read_csv('takehome_assignments.csv')

# Check datatype of variables
def printdtype(data):
          print data.dtypes

printdtype(contacts)
printdtype(assign)
                                
# Adjust data types of time-related variables from object to timestamp
for i in range(3,7):
          contacts[contacts.columns[i]] = pd.to_datetime(contacts[contacts.columns[i]])

# Check if we have successfully changed the datatype - Yay
printdtype(contacts)
printdtype(assign)

# Create a new column named "assignment_ab" in table 'contacts', and map the ab value according to user_id
# Code below will get a warning from python, which is false positive -- Source: stackoverflow question 20625582
def map():
          contacts['assignment_ab'] = ""
          for i in range(len(contacts)):
                    contacts['assignment_ab'][i] = [assign['ab'][assign[assign['id_user'] == contacts['id_guest'][i]].index].values][0][0]
map()

''' Part II. Overview of variables'''

# A first look into datasets
def glance(data):
          print len(data)
          print data.describe()
          print data.ix[0:10,]

glance(contacts)
glance(assign)
print contacts[contacts.columns[0:8]].describe()


''' Part III. Conversion rate and A/B test'''

# Create a table with the number of converted cases of each stages, grouped by ab assignment 
converstat = contacts.groupby('assignment_ab').count()


# Is the treatment group getting more bookings?
def abtest(data):
          data = pd.DataFrame(data)
          observed = data.values
          print observed

          result = chi2_contingency(observed)
          chisq, p = result[:2]
          print 'chisq = {}, p = {}'.format(chisq, p)

d = {'not_converted' : [5009-1094, 4991-1077],'converted' : [1094,1077]}
abtest(d)
############
''' Part IV. Timeline'''
# Create a new column "timespan1"  -- the time between first interation time and first reply time
contacts['timespan1'] = ""
contacts['timespan1'] = contacts['ts_reply_at_first']- contacts['ts_interaction_first']
contacts['timespan1trans'] = ""
for i in range(len(contacts)):
          if not (contacts.timespan1.isnull())[i]:
                    contacts.timespan1trans[i] = contacts.timespan1[i].total_seconds()
          else:
                    contacts.timespan1trans[i] = float('NaN')
contacts.timespan1trans = contacts.timespan1trans.astype(float)


# Create a new column "timespan2"  -- the time between first reply time and first accepted time
contacts['timespan2'] = ""
contacts['timespan2'] = contacts['ts_accepted_at_first']- contacts['ts_reply_at_first']

contacts['timespan2trans'] = ""

for i in range(len(contacts)):
          if not (contacts.timespan2.isnull())[i]:
                    contacts.timespan2trans[i] = contacts.timespan2[i].total_seconds()
          else:
                    contacts.timespan2trans[i] = float('NaN')
contacts.timespan2trans = contacts.timespan2trans.astype(float)


# Create a new column "timespan3"  -- the time between first accepted time and first booking time
contacts['timespan3'] = ""
contacts['timespan3'] = contacts['ts_booking_at']- contacts['ts_accepted_at_first']

contacts['timespan3trans'] = ""
for i in range(len(contacts)):
          if not (contacts.timespan3.isnull())[i]:
                    contacts.timespan3trans[i] = contacts.timespan3[i].total_seconds()
          else:
                    contacts.timespan3trans[i] = float('NaN')
contacts.timespan3trans = contacts.timespan3trans.astype(float)


''' Part V. Channels'''
# check count of channels grouped by assignment
print contacts.groupby(['assignment_ab', 'dim_contact_channel']).size()
grouped = contacts.groupby('assignment_ab')
grouped['m_first_message_length'].agg([np.mean, np.std])
grouped['timespan1trans'].agg([np.mean, np.std])
grouped['timespan2trans'].agg([np.mean, np.std])
grouped['timespan3trans'].agg([np.mean, np.std])

