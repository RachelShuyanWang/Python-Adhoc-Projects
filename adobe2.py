import urllib2
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from BeautifulSoup import BeautifulSoup

# scrape from website using beautiful soup
SOCRurl = "http://wiki.stat.ucla.edu/socr/index.php/SOCR_Data_Dinov_091609_SnP_HomePriceIndex"
socrsoup = BeautifulSoup(urllib2.urlopen(SOCRurl).read())


'''print socrsoup('table')[0]'''
print len(socrsoup('table')[0].findAll('tr')[1].findAll('td'))

# create a list of all the table nodes
templist = []

for j in range(len(socrsoup('table')[0].findAll('tr')[1].findAll('td'))):
          templist.append(socrsoup('table')[0].findAll('tr')[0].findAll('th')[j].text.encode('utf-8'))

for i in range(1, len(socrsoup('table')[0].findAll('tr'))):
          for j in range(len(socrsoup('table')[0].findAll('tr')[1].findAll('td'))):
                    templist.append(socrsoup('table')[0].findAll('tr')[i].findAll('td')[j].text.encode('utf-8'))

# use Numpy to turn list into array
temparray = np.array(templist).reshape((223,23))


# use Pandas to turn array into data frame
# data type object to string/float
socr = pd.DataFrame(temparray[1:,1:], index = temparray[1:,0], columns = temparray[0,1:])

socr.describe()
socr[socr.columns[0]] = socr[socr.columns[0]].astype(str)
socr[socr.columns[1]] = socr[socr.columns[1]].astype(str)

for i in range(2,len(socr.columns)):
          socr[socr.columns[i]] = socr[socr.columns[i]].astype(float)

# create new variable Time to concatenate Year and Month
socr.Time = pd.to_datetime(socr.Year + socr.Month, format="%Y%B")
socr['Time']= pd.Series(socr.Time, index=socr.index)


# plot Composite Index
plt.plot(socr.Time, socr['Composite-10'])
plt.title('Trend of Composite Index Over Time')
plt.ylabel('Composite Index')
plt.xlabel('Time')
plt.savefig('figure1.png')

# plot all 19 regions
for i in range(2, len(socr.columns)-1):
          plt.plot(socr.Time, socr[socr.columns[i]])
plt.savefig('figure2.png')

'''
# regression
est = sm.OLS(socr['Composite-10'], socr.Time)
est =est.fit()
est.summary()

mod1 = sm.ols('socr.Time ~ socr['Composite-10']', data=socr).fit()
'''

def runup(indexlist):
          return max(indexlist)

for i in range(2, len(socr.columns)-1):
          maxindex = 0
          maxi = i
          if runup(socr[socr.columns[i]]) > maxindex:
                    maxindex = runup(socr[socr.columns[i]])
                    maxi = i
          else:
                    pass
print i


def crush(indexlist):
          for i in range(len(indexlist)):
                    minindex = indexlist[0]
                    maxcrush = 0
                    if indexlist[i] < minindex:
                              minindex = indexlist[i]
                    elif indexlist[i] - minindex > maxcrush:
                              maxcrush = indexlist[i] - minindex
                    else:
                              pass
          return maxcrush

for i in range(2, len(socr.columns)-1):
          maxindex = 0
          maxi = i
          if crush(socr[socr.columns[i]]) > maxindex:
                    maxindex = runup(socr[socr.columns[i]])
                    maxi = i
          else:
                    pass
print i






