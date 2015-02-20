# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np; import pandas as pd; import math; import matplotlib.pyplot as plt; import glob;

# <codecell>

pd.__version__

# <codecell>

#FC Loads data files in trips data frame. Run the script in the folder where the drivers' (users') folders are.
trips = []
users = glob.glob('./351*/')
for user in users:
    trip_files = glob.glob('%s/*.csv'%user)
    trips_u = []
    for tf in trip_files:
        trips_u_tf = pd.read_csv( tf )
        trips_u_tf['user'] = int(user.split('/')[1]) * np.ones( len( trips_u_tf ), dtype=int )
        trips_u_tf['trip'] = int(tf.split('/')[2].split('.')[0]) * np.ones( len( trips_u_tf ), dtype=int )
        trips_u.append( trips_u_tf )
    trips_u = pd.concat(  trips_u )
    trips.append( trips_u )
trips = pd.concat( trips )

# <codecell>

trips['time'] = trips.index

# <codecell>

trips = trips.sort(['user','trip','time'])

# <codecell>

#FC I'm choosing an index from 0 to something since the fancier indexing techniques (.loc .ix) I've tried are horribly slow.
trips = trips.reset_index(drop = True )

# <codecell>

#trips = trips.set_index(['user','trip','time'])
#df = trips.loc[3511,26,:]
#trips.xs( (3515,26,1), level=['user','trip','time'])

# <codecell>


# <codecell>

#FC selects user just not to run the code through all of the data
user_n = 3516
n_trips_u = len( set( trips_u['trip'] ) )

# <codecell>

#FC graphs the trips of the chosen user
trips_u = trips[ trips['user']==user_n ]
for t in np.arange(n_trips_u)+1: #trips.ix[user_n,:,:].index.levels[0]: #set(trips_u['trip']): #xrange(1,10,1):
    trips_u_t = trips_u[ trips_u['trip']==t ]
    
    plt.plot( trips_u_t['x'], trips_u_t['y'], linestyle='-') #, marker='*' )
    plt.title('user %d, %d trips'%(user_n, n_trips_u) )

# <codecell>


# <codecell>

#FC calculates the 'signed' inverse radius of the circle through three consecutive points
def sign_curv(p_0,p_1,p_2):
    center = [-9999,-9999]
    sign = np.sign( (p_1[0]-p_0[0])*(p_2[1]-p_1[1]) - (p_2[0]-p_1[0])*(p_1[1]-p_0[1]) )
    
    if sign == 0:
        return 0
    
    else:
        if (p_1[0]-p_0[0]) != 0:
            s_0 = (p_1[1]-p_0[1])/float(p_1[0]-p_0[0])
        
            if (p_2[0]-p_1[0]) != 0:
                s_1 = (p_2[1]-p_1[1])/float(p_2[0]-p_1[0])
            
                center[0] = 0.5*( s_0*s_1*(p_2[1]-p_0[1]) + s_0*(p_1[0]+p_2[0]) - s_1*(p_0[0]+p_1[0]) ) / (s_0-s_1)
            
                if s_0 == 0:
                    center[1] = s_1*center[0] + 0.5*( p_2[1]+p_1[1] - s_1*(p_2[0]+p_1[0]) )
                else:
                    center[1] = -(1/s_0)*( center[0] - 0.5*(p_0[0]+p_1[0]) ) +  0.5*(p_0[1]+p_1[1])
                    
                rad = sqrt( (p_0[0]-center[0])**2 + (p_0[1]-center[1])**2 )
    
                return sign*(1/float(rad))
    
            else:
                center[1] = 0.5*(p_2[1] + p_1[1])
            
                if s_0 == 0:
                    center[0] = 0.5*( p_1[0] + p_0[0] )
                else:
                    center[0] = (1/s_0)*( center[1] - 0.5* (p_1[1]-p_0[1]) ) + 0.5*(p_1[0]-p_0[0]) 
    
                rad = sqrt( (p_0[0]-center[0])**2 + (p_0[1]-center[1])**2 )
            
                return sign*(1/float(rad))
        
        else:
            s_1 = (p_2[1]-p_1[1])/float(p_2[0]-p_1[0])
            
            center[1] = 0.5*(p_1[1] + p_0[1])
            
            if s_1 == 0:
                center[0] = 0.5*(p_2[0] + p_1[0])
            else:
                center[0] = (1/s_1)*( center[1] - 0.5* (p_2[1]-p_1[1]) ) + 0.5*(p_2[0]-p_1[0]) 
    
            rad = sqrt( (p_0[0]-center[0])**2 + (p_0[1]-center[1])**2 )   
        
            return sign*(1/float(rad))

# <codecell>

#FC applies sign_curv on a dataframe with two columns
def df_curv(df):
    p = np.array(df)
    c = [0]
    
    zero = np.array([0,0])
    for i in xrange(1,len(df)-1,1):
        c.append( sign_curv(p[i-1],p[i],p[i+1])*( not ( np.all(p[i]==0) or np.all(p[i+1]==0) ) ) )
    
    c.append(0)
    return pd.DataFrame(c).reindex(df.index)

# <codecell>


# <codecell>

#FC creates a new column in trips
trips['sign_curv'] = df_curv( trips[['x','y']] )

# <codecell>

trips[10000:10010]

# <codecell>


# <codecell>

#FC plot of part of the previous computation
trips_u = trips[ trips['user']==user_n ]
for t in np.arange(n_trips_u)+1: #trips.ix[user_n,:,:].index.levels[0]: #set(trips_u['trip']): #xrange(1,10,1):
    trips_u_t = trips_u[ trips_u['trip']==t ]
    
    plt.plot( np.arange(len(trips_u_t))[:100], trips_u_t['sign_curv'][:100], linestyle='-') #, marker='*' )
    plt.title('user %d, %d trips'%(user_n, n_trips_u) )

# <codecell>


# <codecell>

#FC calculates the norm of the acceleration vector and its perp and norm components based on three consecutive points
def norm_acce(p_0,p_1,p_2):
    p_12 = p_2 - p_1
    p_01 = p_1 - p_0
    
    norm = np.linalg.norm( p_12-p_01 )
    
    norm_p_01_2 = np.dot( p_01, p_01 )
    if norm_p_01_2 != 0:
        norm_p_01 = math.sqrt( norm_p_01_2 )
        dot = np.dot(p_01,p_12)
        norm_perp = np.linalg.norm( p_12 - (1/float(norm_p_01_2))*dot*p_01 )
    
    
        norm_tang = abs( (1/float(norm_p_01_2))*dot - 1 )*norm_p_01
    else:
        norm_tang = norm
        norm_perp = 0

    return (norm, norm_perp, norm_tang)

# <codecell>

#FC applies norm_acce on a dataframe with two columns
def df_acce(df):
    p = np.array(df)
    
    c = [ np.array([0,0,0]) ]
    for i in xrange(1,len(df)-1,1):
        c.append( np.array(norm_acce(p[i-1], p[i], p[i+1]))*( not ( np.all(p[i]==0) or np.all(p[i+1]==0) ) ) )
    
    c.append( np.array([0,0,0]) )
    return pd.DataFrame(c).reindex(df.index)

# <codecell>


# <codecell>

trips[['acce','acce_perp','acce_tang']] = df_acce(trips[['x','y']])

# <codecell>

trips[10000:10010]

# <codecell>

#FC plots some of the above acce components
trips_u = trips[ trips['user']==user_n ]
for t in np.arange(n_trips_u)+1: #trips.ix[user_n,:,:].index.levels[0]: #set(trips_u['trip']): #xrange(1,10,1):
    trips_u_t = trips_u[ trips_u['trip']==t ]
    
    plt.plot( np.arange(len(trips_u_t))[:50], trips_u_t['acce'][:50], linestyle='-') #, marker='*' )
    plt.title('user %d, %d trips'%(user_n, n_trips_u) )

# <codecell>


# <codecell>

#FC calculates the speed based on two consecutive points
def speed(p_0,p_1):
    p_01 = p_1 - p_0
    
    norm = np.linalg.norm( p_01 )

    return norm

# <codecell>

#FC applies speed on a dataframe with two columns
def df_speed(df):
    p = np.array(df)
    
    c = [ 0 ]
    for i in xrange(1,len(df)-1,1):
        c.append( speed(p[i-1], p[i])*( not ( np.all(p[i]==0) ) ) ) 
    
    return pd.DataFrame(c).reindex(df.index)

# <codecell>

trips['speed'] = df_speed(trips[['x','y']])

# <codecell>

trips[10000:10010]

# <codecell>

#FC plots some of the above acce components
trips_u = trips[ trips['user']==user_n ]
for t in np.arange(n_trips_u)+1: #trips.ix[user_n,:,:].index.levels[0]: #set(trips_u['trip']): #xrange(1,10,1):
    trips_u_t = trips_u[ trips_u['trip']==t ]
    
    plt.plot( np.arange(len(trips_u_t))[:50], trips_u_t['speed'][:50], linestyle='-') #, marker='*' )
    plt.title('user %d, %d trips'%(user_n, n_trips_u) )

# <codecell>


