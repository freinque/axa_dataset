#prel_feat_02_write_sql.py
#FC This script was used to read the GPS coordinates .csv files (provided by 
#AXA as a Kaggle challenge) and write derived quantities (speed, acceleration,
#tangent acceleration, perpendicular acceleration and curvature) to a SQL
#database. It is slow for sure, but is good enough for our purposes here.
#
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import glob
import sqlalchemy
import pandas.io.sql

#FC calculates velocity of a series of points
def diff( pos_in ):
    pos = np.array( pos_in )
    pos_minus_1 = np.roll(pos,1)
    
    velo = pos - pos_minus_1
    velo[0] = velo[1] 
    velo[-1] = velo[-2] 

    return velo

#FC calculates the norm of the acceleration vector and its perp and norm components based on three consecutive points, now signed 
def tang( v, a, norm_v): 
    if norm_v != 0:
        dot = np.dot(v,a)
        tang = dot/norm_v
    else:
        tang = -np.linalg.norm( a )

    return tang

#FC applies norm_acce on a dataframe with two columns
def acce_tang( row ):
    
    return tang( row[['x\'','y\'']], row[['x\'\'','y\'\'']], row['speed'] )

def curv( row ):
    if (row['speed'] == 0):
        return 0
    else:
        return (row['x\'']*row['y\'\''] - row['x\'\'']*row['y\''])/(row['speed']**3)


#FC Loads data file names as a pandas data frame and splits them into batches.
trips = glob.glob('/home/freinque/datasets/axa/drivers/*/*')
trips = pd.DataFrame( [ np.array([ int(trip.split('/')[6]), int(trip.split('/')[7].split('.')[0])]) for trip in trips ], columns=['user','trip'])
trips.set_index(['user','trip'], inplace=True)
trips.sort(inplace=True)
print len(trips), ' raw trips in folder'
trip_indices = np.split(trips.index.values,456)

#pandas.io.sql.execute("CREATE DATABASE trips_02",engine) #create db, if needed
#engine = sqlalchemy.create_engine('sqlite:////home/freinque/Desktop/char_sieg/datasets/axa') #if sqlite is preferred
engine = sqlalchemy.create_engine('mysql://root:root@localhost/trips_02')

def starts_with( s, temp ):
    return s[0:len(temp)] == temp

#gathers from the database the indices of the trips already preprocessed
old_users = pd.read_sql_query('SHOW TABLES FROM trips_02', engine)
old_users = old_users['Tables_in_trips_02']
old_users = old_users[ old_users.apply(lambda x:starts_with(x,'user')) ]
old_users = list( old_users.apply( lambda x:int(x[4:]) ) )
old_trip_indices = []
for user in old_users:
    old_user_trips = np.array( pd.read_sql_query('SELECT DISTINCT trip FROM user%d'%user, engine)['trip'] )
    for t in old_user_trips:
        old_trip_indices.append((user,t))
print len(old_trip_indices), ' trips already in database'
old_trip_indices = set(old_trip_indices)

#FC4 writes the proprocessed data of very user as a SQL table
for batch in range(151):
    print batch
    miss_ind = set(trip_indices[batch])-old_trip_indices
    for trip_index in miss_ind:
        print 'user ', trip_index[0], ' trip ', trip_index[1], ' preprocessed, batch ', batch 
        trips_u_tf = pd.read_csv(  '/home/freinque/datasets/axa/drivers/%d/%d.csv'%trip_index )
        trips_u_tf['user'] = trip_index[0] * np.ones( len( trips_u_tf ), dtype=int )
        trips_u_tf['trip'] = trip_index[1] * np.ones( len( trips_u_tf ), dtype=int )
                
        trips_u_tf['time'] = trips_u_tf.index
        trips_u_tf = trips_u_tf.sort(['user','trip','time'])
        trips_u_tf = trips_u_tf.set_index(['user','trip','time'])
                
        grouped_trips_u_tf = trips_u_tf.groupby(level=['user','trip'])

        trips_u_tf[['x\'','y\'']] = grouped_trips_u_tf.transform( diff )
        trips_u_tf['speed'] = trips_u_tf[['x\'','y\'']].apply( np.linalg.norm, axis=1 )
        trips_u_tf['dist'] = grouped_trips_u_tf['speed'].transform( np.cumsum )
        
        grouped_trips_u_tf = trips_u_tf[['x\'','y\'']].groupby(level=['user','trip'])
        
        trips_u_tf[['x\'\'','y\'\'']] = grouped_trips_u_tf.transform( diff )
        trips_u_tf['acce'] = trips_u_tf[['x\'\'','y\'\'']].apply( np.linalg.norm, axis=1 )
        trips_u_tf['tang'] = trips_u_tf.apply( acce_tang, axis=1)
        trips_u_tf['perp'] = np.sqrt( abs(trips_u_tf['acce']**2-trips_u_tf['tang']**2) )*np.sign( trips_u_tf['x\'']*trips_u_tf['y\'\''] - trips_u_tf['x\'\'']*trips_u_tf['y\''] )
        trips_u_tf['curv'] = trips_u_tf.apply( curv, axis=1 )
        
        trips_u_tf = trips_u_tf[['x','y','speed','dist','acce','tang','perp','curv']]
            
        trips_u_tf.to_sql('user%d'%trip_index[0], engine, if_exists='append', index=True)
    
        print 'user ', trip_index[0], ' trip ', trip_index[1], ' preprocessed, batch ', batch 
 


