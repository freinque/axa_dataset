# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

 

import numpy as np; import pandas as pd; import math; import matplotlib.pyplot as plt; import glob;

#FC calculates the speed based on two consecutive points
def speed(p_0,p_1):
    p_01 = p_1 - p_0
    
    norm = np.linalg.norm( p_01 )

    return norm

 

#FC applies speed on a dataframe with two columns
def df_speed(df):
    p = np.array(df)
    
    c = [ 0 ]
    for i in xrange(1,len(df),1):
        c.append( speed(p[i-1], p[i])*( not ( np.all(p[i]==0) ) ) ) 
    
    return pd.DataFrame(c).reindex(df.index)

 

 #FC calculates the norm of the acceleration vector and its perp and norm components based on three consecutive points
#FC3 now signed components
def acce(p_0,p_1,p_2):
    p_12 = p_2 - p_1
    p_01 = p_1 - p_0
    
    norm = np.linalg.norm( p_12-p_01 )
    
    norm_p_01_2 = np.dot( p_01, p_01 )
    if norm_p_01_2 != 0:
        norm_p_01 = math.sqrt( norm_p_01_2 )
        dot = np.dot(p_01,p_12)
        sign = np.sign( p_01[0]*p_12[1] - p_12[0]*p_01[1] )
        perp = sign*np.linalg.norm( p_12 - (1/float(norm_p_01_2))*dot*p_01 )
    
        tang = ( (1/float(norm_p_01_2))*dot - 1 )*norm_p_01
    else:
        tang = norm
        perp = 0

    return (norm, perp, tang)

 

#FC applies norm_acce on a dataframe with two columns
def df_acce(df):
    p = np.array(df)
    
    c = [ np.array([0,0,0]) ]
    for i in xrange(1,len(df)-1,1):
        c.append( np.array(acce(p[i-1], p[i], p[i+1]))*( not ( np.all(p[i]==0) or np.all(p[i+1]==0) ) ) )
    
    c.append( np.array([0,0,0]) )
    return pd.DataFrame(c).reindex(df.index)


#FC calculates the 'signed' inverse radius of the circle through three consecutive points
def sign_curv(p_0,p_1,p_2):
    center = [-9999,-9999]
    p_01 = p_1 - p_0
    p_12 = p_2 - p_1
    
    sign = np.sign( p_01[0]*p_12[1] - p_12[0]*p_01[1] )
    
    if sign == 0:
        return 0
    
    else:
        if p_01[0] != 0:
            s_0 = p_01[1]/float(p_01[0])
        
            if p_12[0] != 0:
                s_1 = p_12[1]/float(p_12[0])
            
                center[0] = 0.5*( s_0*s_1*(p_2[1]-p_0[1]) + s_0*(p_1[0]+p_2[0]) - s_1*(p_0[0]+p_1[0]) ) / (s_0-s_1)
            
                if s_0 == 0:
                    center[1] = s_1*center[0] + 0.5*( p_2[1]+p_1[1] - s_1*(p_2[0]+p_1[0]) )
                else:
                    center[1] = -(1/s_0)*( center[0] - 0.5*(p_0[0]+p_1[0]) ) +  0.5*(p_0[1]+p_1[1])
                    
                rad = math.sqrt( (p_0[0]-center[0])**2 + (p_0[1]-center[1])**2 )
    
                return sign*(1/float(rad))
    
            else:
                #FC2 stupid error fixed
                center[1] = 0.5*(p_2[1] + p_1[1])
            
                if s_0 == 0:
                    center[0] = 0.5*( p_1[0] + p_0[0] )
                else:
                    center[0] = s_0*( 0.5* (p_0[1]+p_1[1]) - center[1] ) + 0.5*(p_0[0]+p_1[0]) 
    
                rad = math.sqrt( (p_0[0]-center[0])**2 + (p_0[1]-center[1])**2 )
            
                return sign*(1/float(rad))
        
        else:
            #FC2 stupid error fixed
            s_1 = p_12[1]/float(p_12[0])
            
            center[1] = 0.5*(p_1[1] + p_0[1])
            
            if s_1 == 0:
                center[0] = 0.5*(p_2[0] + p_1[0])
            else:
                center[0] = s_1*( 0.5*(p_1[1]+p_2[1]) - center[1] ) + 0.5*(p_1[0]+p_2[0]) 
    
            rad = math.sqrt( (p_0[0]-center[0])**2 + (p_0[1]-center[1])**2 )   
        
            return sign*(1/float(rad))

 

#FC applies sign_curv on a dataframe with two columns
def df_curv(df):
    p = np.array(df)
    c = [0]
    
    zero = np.array([0,0])
    for i in xrange(1,len(df)-1,1):
        c.append( sign_curv(p[i-1],p[i],p[i+1])*( not ( np.all(p[i]==0) or np.all(p[i+1]==0) ) ) )
    
    c.append(0)
    return pd.DataFrame(c).reindex(df.index)


#pd.__version__

#FC Loads data files in trips data frame. Run the script in the folder where the drivers' (users') folders are.

users = glob.glob('./drivers/*/')
users = np.array([ int(user.split('/')[2]) for user in users ])
users = np.sort(users)
users = np.split(users,228)

#FC4 performed with xrange(0,5,1) and then xrange(5,10,1)

for c in xrange(0,10,1):#range(len(users)):
    for user in users[c]:
        trip_files = glob.glob('./drivers/%i/*.csv'%user)
        trips_u = []
   
        for tf in trip_files:
            trips_u_tf = pd.read_csv( tf )
            trips_u_tf['user'] = user * np.ones( len( trips_u_tf ), dtype=int )
            trips_u_tf['trip'] = int(tf.split('/')[3].split('.')[0]) * np.ones( len( trips_u_tf ), dtype=int )
            trips_u.append( trips_u_tf )
    
        trips_u = pd.concat(  trips_u )

 

        trips_u['time'] = trips_u.index

 

        trips_u = trips_u.sort(['user','trip','time'])

 

    #FC I'm choosing an index from 0 to something since the fancier indexing techniques (.loc .ix) I've tried are horribly slow.
        trips_u = trips_u.reset_index(drop = True )

 


#FC creates a new column in trips
        trips_u['sign_curv'] = df_curv( trips_u[['x','y']] )

 


 

        trips_u[['acce','acce_perp','acce_tang']] = df_acce(trips_u[['x','y']])

 



        trips_u['speed'] = df_speed(trips_u[['x','y']])

        grouped_trips = trips.groupby(['user','trip']) 
        trips['dist'] = grouped_trips['speed'].apply( np.cumsum )
        
        if user == 1:
            trips_u.to_csv('./trips.csv', index=False, mode='w')
        else:
            trips_u.to_csv('./trips.csv', index=False, header=False, mode='a')

    
        print user, ' printed'
 


 

 

