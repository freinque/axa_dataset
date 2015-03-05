# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np; import pandas as pd; import math; import matplotlib.pyplot as plt; import glob;

# <codecell>

pd.__version__

# <codecell>

trips = pd.read_csv( './trips.csv' )

# <codecell>


# <codecell>

#FC calculates a df made of averages/max on the columns of each trip

# <codecell>

f = {'sign_curv': np.mean, 'acce':np.mean, 'acce_perp':np.mean, 'acce_tang':np.mean, 'speed':np.mean}

# <codecell>

grouped_trips = trips.groupby(['user','trip'])

# <codecell>

#FC calculates the average ['sign_curv','acce','acce_perp','acce_tang','speed'] for each trip 
averages = grouped_trips.agg(f)
averages['trip_dura'] = grouped_trips['time'].apply(np.max)
averages['trip_dist'] = grouped_trips['dist'].apply(np.max)

# <codecell>

max_trip_dura = np.max(averages['trip_dura'])
max_trip_dist = np.max(averages['trip_dist'])

# <codecell>

#FC just look at a few things, not important
max_trip_dist, max_trip_dura

# <codecell>

averages['trip_dist'].argmax()

# <codecell>

user_n = 126
trips_u = trips[ trips['user'] == user_n ]
n_trips_u = len( set( trips_u['trip'] ) )
averages_u = averages.loc[user_n, :]
averages_u[150:155]

# <codecell>


# <codecell>

#FC function that will plot an histogram from the output data of np.histogram
def plot_hist(np_hist):
    return plt.bar(np_hist[1][:-1],np_hist[0]/float( np.sum(np_hist[0]) ), width=np_hist[1][1]-np_hist[1][0] )
def plot_hist_nn(np_hist):
    return plt.bar(np_hist[1][:-1],np_hist[0] , width=np_hist[1][1]-np_hist[1][0] )

# <codecell>


# <codecell>


# <codecell>


# <codecell>

##########FC here I start looking for outliers in the trips taken by our chosen user, comparing them only to the average behaviour of our user

# <codecell>

outl_trip = 108

# <codecell>

#FC for each trip, plot a acce_perp distribution, plots a chosen one
hist_u_acce_perp = []
for tr in np.arange(1,n_trips_u+1,1):
    hist_u_acce_perp.append( np.histogram(grouped_trips.get_group((user_n,tr))['acce_perp'],bins=np.linspace(-3,3,num=100))[0] )
hist_u_acce_perp = np.array(hist_u_acce_perp)
hist_u_acce_perp = [ line / float(np.sum(line)) for line in hist_u_acce_perp ]
plot_hist([ hist_u_acce_perp[outl_trip-1], np.linspace(-3,3,num=100)])

# <codecell>

#FC calculates the mean distribution over the trips of our chosen user, plots it
hist_u_acce_perp_mean = np.mean(hist_u_acce_perp,axis=0)
plot_hist([ hist_u_acce_perp_mean, np.linspace(-3,3,num=100)])

# <codecell>

#FC translates the distributions by the mean distribution, plots one
hist_u_acce_perp_tran = np.array( [ line - hist_u_acce_perp_mean for line in hist_u_acce_perp ] )
plot_hist([ hist_u_acce_perp_tran[outl_trip-1], np.linspace(-3,3,num=100)])

# <codecell>

#FC calculates the covariance matrix of the set of trip distributions
hist_u_acce_perp_cov = np.cov( hist_u_acce_perp_tran.T )

# <codecell>

#FC calculates the 'distance squared from the average' of each trip in the covariance metric, plots a histogram of it
hist_u_acce_perp_norm = np.array([ np.dot( line , np.dot(hist_u_acce_perp_cov, line) ) for line in hist_u_acce_perp_tran ])
plot_hist_nn( np.histogram(hist_u_acce_perp_norm, np.linspace(0,0.01,num=100)) )

# <codecell>

#FC gives the trip number of the trips that are farthest from average
hist_u_acce_perp_norm.argsort()[-5:][::-1]+1

# <codecell>


# <codecell>

#FC repeats the procedure for acce_tang
hist_u_acce_tang = []
for tr in np.arange(1,n_trips_u+1,1):
    hist_u_acce_tang.append( np.histogram(grouped_trips.get_group((user_n,tr))['acce_tang'],bins=np.linspace(-3,3,num=100))[0] )
hist_u_acce_tang = np.array(hist_u_acce_tang)
hist_u_acce_tang = [ line / float(np.sum(line)) for line in hist_u_acce_tang ]
plot_hist([ hist_u_acce_tang[outl_trip-1], np.linspace(-3,3,num=100)])

# <codecell>

hist_u_acce_tang_mean = np.mean(hist_u_acce_tang,axis=0)
plot_hist([ hist_u_acce_tang_mean, np.linspace(-3,3,num=100)])

# <codecell>

hist_u_acce_tang_tran = np.array( [ line - hist_u_acce_tang_mean for line in hist_u_acce_tang ] )
plot_hist([ hist_u_acce_tang_tran[outl_trip-1], np.linspace(-3,3,num=100)])

# <codecell>

hist_u_acce_tang_cov = np.cov( hist_u_acce_tang_tran.T )
hist_u_acce_tang_norm = np.array([ np.dot( line , np.dot(hist_u_acce_tang_cov, line) ) for line in hist_u_acce_tang_tran ])
plot_hist_nn( np.histogram(hist_u_acce_tang_norm, np.linspace(0,0.01,num=100)) )

# <codecell>

hist_u_acce_tang_norm.argsort()[-5:][::-1]+1

# <codecell>


# <codecell>

hist_u_speed = []
for tr in np.arange(1,n_trips_u+1,1):
    hist_u_speed.append( np.histogram(grouped_trips.get_group((user_n,tr))['speed'],bins=np.linspace(0,40,num=100))[0] )
hist_u_speed = np.array(hist_u_speed)
hist_u_speed = [ line / float(np.sum(line)) for line in hist_u_speed ]
plot_hist([ hist_u_speed[outl_trip-1], np.linspace(0,40,num=100)])

# <codecell>

hist_u_speed_mean = np.mean(hist_u_speed,axis=0)
plot_hist([ hist_u_speed_mean, np.linspace(0,40,num=100)])

# <codecell>

hist_u_speed_tran = np.array( [ line - hist_u_speed_mean for line in hist_u_speed ] )
hist_u_speed_cov = np.cov( hist_u_speed_tran.T )
hist_u_speed_norm = np.array([ np.dot( line , np.dot(hist_u_speed_cov, line) ) for line in hist_u_speed_tran ])
plot_hist_nn( np.histogram(hist_u_speed_norm, np.linspace(0,0.01,num=100)) )

# <codecell>

hist_u_speed_norm.argsort()[-5:][::-1]+1

# <codecell>


# <codecell>

hist_u_sign_curv = []
for tr in np.arange(1,n_trips_u+1,1):
    hist_u_sign_curv.append( np.histogram(grouped_trips.get_group((user_n,tr))['sign_curv'],bins=np.linspace(-0.07,0.07,num=100))[0] )
hist_u_sign_curv = np.array(hist_u_sign_curv)
hist_u_sign_curv = [ line / float(np.sum(line)) for line in hist_u_sign_curv ]
plot_hist([ hist_u_sign_curv[outl_trip], np.linspace(-0.07,0.07,num=100)])

# <codecell>

hist_u_sign_curv_mean = np.mean(hist_u_sign_curv,axis=0)
plot_hist([ hist_u_sign_curv_mean, np.linspace(-0.07,0.07,num=100)])

# <codecell>

hist_u_sign_curv_tran = np.array( [ line - hist_u_sign_curv_mean for line in hist_u_sign_curv ] )
hist_u_sign_curv_cov = np.cov( hist_u_sign_curv_tran.T )
hist_u_sign_curv_norm = np.array([ np.dot( line , np.dot(hist_u_sign_curv_cov, line) ) for line in hist_u_sign_curv_tran ])
plot_hist_nn( np.histogram(hist_u_sign_curv_norm, np.linspace(0,0.01,num=100)) )

# <codecell>

hist_u_sign_curv_norm.argsort()[-5:][::-1]+1

# <codecell>


# <codecell>


# <codecell>

#FC choose a user and plot trip coordinates, just for visual comparison with outlier proposed below
trips_u = trips[ trips['user'] == user_n ]
n_trips_u = len( set( trips_u['trip'] ) )
#FC graphs the trips of the chosen user
trips_u = trips[ trips['user']==user_n ]
for t in np.arange(n_trips_u)+1: #trips.ix[user_n,:,:].index.levels[0]: #set(trips_u['trip']): #xrange(1,10,1):
    trips_u_t = trips_u[ trips_u['trip']==t ]
    
    plt.plot( trips_u_t['x'], trips_u_t['y'], linestyle='-') #, marker='*' )
    plt.title('user %d, %d trips'%(user_n, n_trips_u) )

# <codecell>

#FC plot a particular trip that we suspect to be an outlier trip
tr=outl_trip
trips_u_t = trips_u[ trips_u['trip']==tr ]
    
plt.plot( trips_u_t['x'][:200], trips_u_t['y'][:200], linestyle='-') #, marker='*' )
plt.title('user %d, %d trips'%(user_n, n_trips_u) )

# <codecell>

weird_trip = grouped_trips.get_group((user_n,outl_trip))

# <codecell>

weird_trip[:10]

# <codecell>


# <codecell>

################FC starting to compare the behaviour of the chosen user to that of the average user

# <codecell>

#FC trip distance distribution for the chosen user
hist_u_trip_dist = np.histogram(averages_u['trip_dist'],bins=np.linspace(0,30000,num=100))#max_trip_dist,num=100))
plot_hist(hist_u_trip_dist);

# <codecell>

#FC trip distance distribution for all users
hist_trip_dist = np.histogram(averages['trip_dist'],bins=np.linspace(0,30000,num=100))#max_trip_dist,num=100))
plot_hist(hist_trip_dist)

# <codecell>

#FC same for trip duration
hist_u_trip_dura = np.histogram(averages_u['trip_dura'],bins=np.linspace(0,max_trip_dura,num=100))
plot_hist(hist_u_trip_dura)

# <codecell>

hist_trip_dura = np.histogram(averages['trip_dura'],bins=np.linspace(0,max_trip_dura,num=100))
plot_hist(hist_trip_dura)

# <codecell>

#FC same for speed
hist_u_speed = np.histogram(trips_u['speed'],bins=np.linspace(0,40,num=100))
plot_hist(hist_u_speed)

# <codecell>

hist_speed = np.histogram(trips['speed'],bins=np.linspace(0,40,num=100))
plot_hist(hist_speed)

# <codecell>

#FC same for acce
hist_u_acce = np.histogram(trips_u['acce'],bins=np.linspace(0,6,num=100))
plot_hist(hist_u_acce)

# <codecell>

hist_acce = np.histogram(trips['acce'],bins=np.linspace(0,6,num=100))
plot_hist(hist_acce)

# <codecell>

#FC same for sign_curv
hist_u_sign_curv = np.histogram(trips_u['sign_curv'],bins=np.linspace(-0.05,0.05,num=100))
plot_hist(hist_u_sign_curv)

# <codecell>

hist_sign_curv = np.histogram(trips['sign_curv'],bins=np.linspace(-0.05,0.05,num=100))
plot_hist(hist_sign_curv)

# <codecell>

#FC same for acce_tang
hist_u_acce_tang = np.histogram(trips_u['acce_tang'],bins=np.linspace(-3,3,num=100))
plot_hist(hist_u_acce_tang)

# <codecell>

hist_acce_tang = np.histogram(trips['acce_tang'],bins=np.linspace(-3,3,num=100))
plot_hist(hist_acce_tang)

# <codecell>

#FC same for acce_perp
hist_u_acce_perp = np.histogram(trips_u['acce_perp'],bins=np.linspace(-3,3,num=100))
plot_hist(hist_u_acce_perp)

# <codecell>

hist_acce_perp = np.histogram(trips['acce_perp'],bins=np.linspace(-3,3,num=100))
plot_hist(hist_acce_perp)

# <codecell>


