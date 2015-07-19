#FC This script explores some features that can be used to try classiy a 
#given car trip made by an unknown user. It could be seen as a necessary
#first step towards finding a unique identifier for every driver in the database.
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import sqlalchemy
import pandas.io.sql
import sklearn.mixture 
import sklearn.decomposition

#FC start a sqlalchemy engine that connects to the trips_02 mysql database 
#engine = sqlalchemy.create_engine('sqlite:///trips.db') #if sqlite
engine = sqlalchemy.create_engine('mysql://root:root@localhost/trips_02')

#FC abs and then mean
def mean_circ_abs( values ):
    abs_values = abs(values)
    return np.mean( abs_values )

#FC function that will plot an histogram from the output data of np.histogram
def plot_hist(np_hist):
    return plt.bar(np_hist[1][:-1],np_hist[0]/float( np.sum(np_hist[0]) ), width=np_hist[1][1]-np_hist[1][0] )
def plot_hist_nn(np_hist):
    return plt.bar(np_hist[1][:-1],np_hist[0] , width=np_hist[1][1]-np_hist[1][0] )

#FC function that returns a numpy histogram as a DataFrame
def dist( col, lo_bound, hi_bound, n_bins ):
    histo = np.histogram( col, bins=np.linspace(lo_bound,hi_bound,num=n_bins),density=True)
    df = pd.DataFrame( histo[0], index=histo[1][:-1] )
    df.columns = ['prop']
    return df

#FC splits an array in n (approx) equal length sub-arrays and sums each of them
def split_sum( arr, n ):
    nparr = np.array(arr)
    nparr = np.array_split(nparr, n)
    df = pd.DataFrame([ np.sum(x) for x in nparr],columns=['prop'])
    df.index.name = 'bin'
    return df

def split_min( arr, n ):
    nparr = np.array(arr)
    nparr = np.array_split(nparr, n)
    df = pd.DataFrame([ np.min(x) for x in nparr],columns=['prop'])
    df.index.name = 'bin'
    return df

#FC takes the exponential of an array and then returns the proportion of each term
def exp_prop( v ):
   npv = np.array(v)
   npv = np.exp(v)
   return npv/float(np.sum(npv))

#FC finds indices of all the trips in the unprocessed .csv database, and splits them into batches
trips = glob.glob('/home/freinque/datasets/axa/drivers/*/*')
trips = pd.DataFrame( [ np.array([ int(trip.split('/')[6]), int(trip.split('/')[7].split('.')[0])]) for trip in trips ], columns=['user','trip'])
trips.set_index(['user','trip'], inplace=True)
trips.sort(inplace=True)
print len(trips), ' raw trips in folder'
trip_indices = np.split(trips.index.values,456)

#############################################################################

#FC computes the relevant averages on the preprocessed data, and writes it to the averages table
average_indices = pd.read_sql_query('SELECT user, trip FROM averages', engine)
average_indices.set_index(['user','trip'],inplace=True)
average_indices.sort(inplace=True)
old_average_indices = set(average_indices.index.values)

f = { 'speed':np.mean, 'acce':np.mean, 'perp':mean_circ_abs, 'tang':mean_circ_abs, 'curv': mean_circ_abs}

for batch in range(151):
    print batch
    miss_ind = set(trip_indices[batch])-old_average_indices
    for trip_index in miss_ind: #trip_indices[batch]:
        trip = pd.read_sql_query('SELECT * FROM user%d WHERE trip = %d'%trip_index, engine)
    
        #calculates averages of the preprocessed data over each trip
        grouped_trip = trip.groupby(['user','trip'])
    
        #FC calculates the average ['curv','acce','perp','tang','speed'] for each trip 
        trip_averages = grouped_trip.agg(f)
        trip_averages['trip_dura'] = grouped_trip['time'].agg(np.max)
        trip_averages['trip_dist'] = grouped_trip['dist'].agg(np.max)
    
        trip_averages.to_sql('averages', engine, if_exists='append', index=True)
        print trip_index, ' averages calculated, batch ', batch

#################

#FC Calculating the (histogram estimator) distributions of the speeds for each trip
speed_dist_indices = pd.read_sql_query('SELECT user, trip FROM speed_dist', engine)
speed_dist_indices.set_index(['user','trip'],inplace=True)
speed_dist_indices.sort(inplace=True)
old_speed_dist_indices = set(speed_dist_indices.index.values)

for batch in range(151):
    print batch
    miss_ind = set(trip_indices[batch])-old_speed_dist_indices
    for trip_index in miss_ind: #trip_indices[batch]:
        trip = pd.read_sql_query('SELECT user, trip, time, speed FROM user%d WHERE trip = %d'%trip_index, engine)
        trip.set_index(['user','trip','time'],inplace=True)
        grouped_trip = trip.groupby(level=['user','trip'])
        
        trip_speed_dist = grouped_trip['speed'].apply( dist, 0,40,40 ) 
        trip_speed_dist.index.rename(['user','trip','speed_bin_lb'], inplace=True)

        trip_speed_dist.to_sql('speed_dist', engine, if_exists='append', index=True)
        print trip_index, ' speed distribution calculated, batch ', batch

#FC Calculating the (histogram estimator) distributions of the norm of the acceleration for each trip
acce_dist_indices = pd.read_sql_query('SELECT user, trip FROM acce_dist', engine)
acce_dist_indices.set_index(['user','trip'],inplace=True)
acce_dist_indices.sort(inplace=True)
old_acce_dist_indices = set(acce_dist_indices.index.values)

for batch in range(151):
    print batch
    miss_ind = set(trip_indices[batch])-old_acce_dist_indices
    for trip_index in miss_ind:
        trip = pd.read_sql_query('SELECT user, trip, time, acce FROM user%d WHERE trip = %d'%trip_index, engine)
        trip.set_index(['user','trip','time'],inplace=True)
        grouped_trip = trip.groupby(level=['user','trip'])
        
        trip_acce_dist = grouped_trip['acce'].apply( dist, 0,8,40 ) 
        trip_acce_dist.index.rename(['user','trip','acce_bin_lb'], inplace=True)

        trip_acce_dist.to_sql('acce_dist', engine, if_exists='append', index=True)
        print trip_index, ' acce distribution calculated, batch ', batch

#FC tang distributions
tang_dist_indices = pd.read_sql_query('SELECT user, trip FROM tang_dist', engine)
tang_dist_indices.set_index(['user','trip'],inplace=True)
tang_dist_indices.sort(inplace=True)
old_tang_dist_indices = set(tang_dist_indices.index.values)

for batch in range(151):
    print batch
    miss_ind = set(trip_indices[batch])-old_tang_dist_indices
    for trip_index in miss_ind:
        trip = pd.read_sql_query('SELECT user, trip, time, tang FROM user%d WHERE trip = %d'%trip_index, engine)
        trip.set_index(['user','trip','time'],inplace=True)
        grouped_trip = trip.groupby(level=['user','trip'])
        
        trip_tang_dist = grouped_trip['tang'].apply( dist, -8,8,40 ) 
        trip_tang_dist.index.rename(['user','trip','tang_bin_lb'], inplace=True)

        trip_tang_dist.to_sql('tang_dist', engine, if_exists='append', index=True)
        print trip_index, ' tang distribution calculated, batch ', batch

#FC perp distributions
perp_dist_indices = pd.read_sql_query('SELECT user, trip FROM perp_dist', engine)
perp_dist_indices.set_index(['user','trip'],inplace=True)
perp_dist_indices.sort(inplace=True)
old_perp_dist_indices = set(perp_dist_indices.index.values)

for batch in range(151):
    print batch
    miss_ind = set(trip_indices[batch])-old_perp_dist_indices
    for trip_index in miss_ind:
        trip = pd.read_sql_query('SELECT user, trip, time, perp FROM user%d WHERE trip = %d'%trip_index, engine)
        trip.set_index(['user','trip','time'],inplace=True)
        grouped_trip = trip.groupby(level=['user','trip'])
        
        trip_perp_dist = grouped_trip['perp'].apply( dist, -8,8,40 ) 
        trip_perp_dist.index.rename(['user','trip','perp_bin_lb'], inplace=True)

        trip_perp_dist.to_sql('perp_dist', engine, if_exists='append', index=True)
        print trip_index, ' perp distribution calculated, batch ', batch

#FC curv distributions
curv_dist_indices = pd.read_sql_query('SELECT user, trip FROM curv_dist', engine)
curv_dist_indices.set_index(['user','trip'],inplace=True)
curv_dist_indices.sort(inplace=True)
old_curv_dist_indices = set(curv_dist_indices.index.values)

for batch in range(151):
    print batch
    miss_ind = set(trip_indices[batch])-old_curv_dist_indices
    for trip_index in miss_ind:
        trip = pd.read_sql_query('SELECT user, trip, time, curv FROM user%d WHERE trip = %d'%trip_index, engine)
        trip.set_index(['user','trip','time'],inplace=True)
        grouped_trip = trip.groupby(level=['user','trip'])
        
        trip_curv_dist = grouped_trip['curv'].apply( dist, -0.1,0.1,40 ) 
        trip_curv_dist.index.rename(['user','trip','curv_bin_lb'], inplace=True)

        trip_curv_dist.to_sql('curv_dist', engine, if_exists='append', index=True)
        print trip_index, ' curv distribution calculated, batch ', batch


#########################################################################################
#FC load the averages as a DataFrame
averages = pd.read_sql_query('SELECT * FROM averages', engine)
averages = averages.set_index(['user','trip'])

#FC just deleting from the database a bunch (<2%) of 'obvious outliers', not so important
#In many cases, the GPS coord makes a big jump (I'm catching them by looking at maxima of averages acce)
#or the trip just looks like a ramdom jitter mess so they're not likely to be useful here
analysed_trips = pd.DataFrame( (averages['acce']<7) & (averages['trip_dist']>30) & (averages['speed']>0.5), columns=['analysed'])
analysed_trips.to_sql('analysed_trips', engine, if_exists='replace', index=True)

#FC load the speed distributions as a DataFrame
#speed_dist = pd.read_sql_query('SELECT * FROM speed_dist', engine)
#speed_dist.set_index(['user','trip'],inplace=True)
#speed_dist = speed_dist.loc[ analysed_indices ]
#plot_hist([ np.array(speed_dist.loc[1,1]['prop'])[:-1], np.array(speed_dist.loc[1,1]['speed_bin_lb']) ])

#FC load the acce distributions as a DataFrame
#acce_dist = pd.read_sql_query('SELECT * FROM acce_dist', engine)
#acce_dist.set_index(['user','trip'],inplace=True)
#acce_dist = acce_dist.loc[ analysed_indices ]
#plot_hist([ np.array(acce_dist.loc[1,1]['prop'])[:-1], np.array(acce_dist.loc[1,1]['acce_bin_lb']) ])

#FC load the tang distributions as a DataFrame
#tang_dist = pd.read_sql_query('SELECT * FROM tang_dist', engine)
#tang_dist.set_index(['user','trip'],inplace=True)
#tang_dist = tang_dist.loc[ analysed_indices ]
#plot_hist([ np.array(tang_dist.loc[1,1]['prop'])[:-1], np.array(tang_dist.loc[1,1]['tang_bin_lb']) ])

#FC load the perp distributions as a DataFrame
#perp_dist = pd.read_sql_query('SELECT * FROM perp_dist', engine)
#perp_dist.set_index(['user','trip'],inplace=True)
#perp_dist = perp_dist.loc[ analysed_indices ]
#plot_hist([ np.array(perp_dist.loc[1,1]['prop'])[:-1], np.array(perp_dist.loc[1,1]['parp_bin_lb']) ])

#FC load the curv distributions as a DataFrame
#curv_dist = pd.read_sql_query('SELECT * FROM curv_dist', engine)
#curv_dist.set_index(['user','trip'],inplace=True)
#curv_dist = curv_dist.loc[ analysed_indices ]
#plot_hist([ np.array(curv_dist.loc[1,1]['prop'])[:-1], np.array(curv_dist.loc[1,1]['curv_bin_lb']) ])



#########################################################################
#FC here I start looking for outliers in the trips taken by our chosen user, comparing them only to the average behaviour of our user

# choosing an arbitrary user
users = pd.read_sql_query('SELECT user FROM analysed_trips', engine)
user = np.random.choice(users['user'])
# recovering the trips to be analysed from the database
user_trips = pd.read_sql_query('SELECT * FROM analysed_trips WHERE user = %d'%user, engine)
user_trips.set_index(['user','trip'], inplace=True)
user_trips = user_trips[ user_trips['analysed']==1 ]
# the types of distributions that will be used here 
types = ['speed','acce','tang','perp','curv'] 

#FC load the distributions of user as a list of DataFrames
user_dist = []
for type_ in types:
    user_dist_type_ = pd.read_sql_query('SELECT * FROM %s_dist WHERE user = %d '%(type_,user), engine)
    user_dist_type_.set_index(['user','trip'],inplace=True)
    user_dist_type_ = user_dist_type_.loc[ user_trips.index ]
    user_dist_type_ = user_dist_type_.groupby(level=['user','trip'])['prop'].apply(split_sum,3)

    user_dist_type__mean = user_dist_type_.groupby(level=['bin']).apply(np.mean)

    #user_dist_type_.set_index(['bin'],append=True,inplace=True)
    user_dist_type_ = user_dist_type_.groupby(level=['user','trip']).apply(lambda x: x - user_dist_type__mean )

    user_dist_type_.drop(0,axis=0,level=2,inplace=True) #should drop a component, since they all 'sum to 0'
    user_dist_type_ = user_dist_type_.groupby(level=['user']).apply(lambda x: x.reset_index().pivot(index='trip',columns='bin',values='prop') )
    user_dist_type_.rename(columns=lambda x:type_+'_resid_'+str(x), inplace=True)
    
    user_dist.append( user_dist_type_ )


user_averages = pd.read_sql_query('SELECT * FROM averages WHERE user = %d'%user, engine)
user_averages = user_averages.set_index(['user','trip'])
user_averages = user_averages.loc[ user_trips.index ] 
#user_averages.drop(['acce','curv','trip_dura','speed'], axis=1, inplace=True)

#user_feat = user_averages
user_feat = user_averages.join( user_dist )
#user_feat = user_dist[0]
#user_feat = user_dist[0].join( user_dist[1:] )

#FC performs a pca reduction (projection) from the space of features above 
pca = sklearn.decomposition.PCA(n_components=None, copy=True, whiten=False)
pca.fit( user_feat )

user_feat_reduced = pca.transform( user_feat )
user_feat_reduced = pd.DataFrame( user_feat_reduced, index=user_feat.index)[range(4)]
#plt.plot(user_feat_reduced[0],user_feat_reduced[1], linestyle='', marker='.')

#FC At this point we notice that for a given user, the distributions of
#['speed','acce','tang','perp','curv'] seem distributed normally, but this
#is not the case when we include the average quantities. It seems natural
#to then consider a normal mixture model instead. 

#Also, sklearn.mixture.GMM.fit seems to be using kmeans for initialization 
#of the means, so we don't have to worry too much about this step

#FC the BIC or AIC criteria don't seem to prevent overfitting very well here,
models = range(0,7)
bics = []
mixture_models = []
for model in models:
    if model < 2: 
        mixture_model = sklearn.mixture.GMM(n_components=1+model, covariance_type='full') 
    else:
        mixture_model = sklearn.mixture.GMM(n_components=model-1, covariance_type='diag') 

    mixture_model.fit( np.array(user_feat_reduced) )
    bic = mixture_model.bic( np.array(user_feat_reduced) )
    
    bics.append( bic )
    mixture_models.append( mixture_model )

print bics
best_model = np.argmin(bics)
if best_model == 0:
    print 'BIC best with 1 full GMM compoment '
else:
    print 'BIC best with diag GMM model with ', best_model, ' components'

best_model = 0
log_likelihoods = pd.Series( mixture_models[best_model].score( np.array(user_feat_reduced) ), index=user_averages.index )
log_likelihoods.sort(ascending=True)
#plot_hist_nn( np.histogram(log_likelihoods, np.linspace(np.min(log_likelihoods),np.max(log_likelihoods),num=100)) )

outlier_index = log_likelihoods.index[0]
outlier_trip = pd.read_sql_query('SELECT * FROM user%d WHERE trip = %d'%outlier_index, engine)
averages_outlier_trip = user_averages.loc[outlier_index]
print averages_outlier_trip
#plt.plot(outlier_trip['x'],outlier_trip['y'])



##########FC here I start comparing the behaviors of two randomly chosen users
#and how well the above GMM can differentiate their trips

analysed_users = pd.read_sql_query('SELECT user FROM analysed_trips', engine)
users = np.sort(np.random.choice(analysed_users['user'], size=2))
print 'analysed users ', users

user_feats = []
for u in range(len(users)):
    # recovering the trips to be analysed from the database
    user_trips = pd.read_sql_query('SELECT * FROM analysed_trips WHERE user = %d'%users[u], engine)
    user_trips.set_index(['user','trip'], inplace=True)
    user_trips = user_trips[ user_trips['analysed']==1 ]
    types = ['speed','acce']#,'tang','perp','acce']#,'curv'] 'acce',

    #FC load the distributions of user as a list of DataFrames
    user_dist = []
    for type_ in types:
        user_dist_type_ = pd.read_sql_query('SELECT * FROM %s_dist WHERE user = %d '%(type_,users[u]), engine)
        user_dist_type_.set_index(['user','trip'],inplace=True)
        user_dist_type_ = user_dist_type_.loc[ user_trips.index ]
        user_dist_type_ = user_dist_type_.groupby(level=['user','trip'])['prop'].apply(split_sum,20)

        user_dist_type__mean = user_dist_type_.groupby(level=['bin']).apply(np.mean)

        #user_dist_type_.set_index(['bin'],append=True,inplace=True)
        user_dist_type_ = user_dist_type_.groupby(level=['user','trip']).apply(lambda x: x )#- user_dist_type__mean )

        user_dist_type_.drop(0,axis=0,level=2,inplace=True) #should drop a component, since they all 'sum to 1'
        user_dist_type_ = user_dist_type_.groupby(level=['user']).apply(lambda x: x.reset_index().pivot(index='trip',columns='bin',values='prop') )
        user_dist_type_.rename(columns=lambda x:type_+'_resid_'+str(x), inplace=True)
    
        user_dist.append( user_dist_type_ )


    user_averages = pd.read_sql_query('SELECT * FROM averages WHERE user = %d'%users[u], engine)
    user_averages = user_averages.set_index(['user','trip'])
    user_averages = user_averages.loc[ user_trips.index ] 
    #user_averages.drop(['trip_dura'], axis=1, inplace=True)

    #user_feat = user_averages
    #user_feat = user_averages.join( user_dist )
    #user_feat = user_dist[0]
    user_feat = user_dist[0].join( user_dist[1:] )

    user_feats.append( user_feat )

users_feat = pd.concat( user_feats )

pca = sklearn.decomposition.PCA(n_components=None, copy=True, whiten=False)
pca.fit( users_feat ) #.loc[users[0]]
users_feat_reduced = pd.DataFrame( pca.transform( users_feat ), index=users_feat.index)[range(6)]
#plt.plot(users_feat_reduced.loc[users[0]][0],users_feat_reduced.loc[users[0]][1], 'bs',users_feat_reduced.loc[users[1]][0],users_feat_reduced.loc[users[1]][1], 'gs')

models = range(7) # full gmm with 1 and 2 comps, diag gmm with 1,2,3,... comps
k=100 # fold cross-validation
users_feat_reduced_perm = users_feat_reduced.loc[ np.random.permutation(users_feat_reduced.index) ]
split_indices = np.array_split( users_feat_reduced_perm, k )

log_likelihoods = []
log_likelihoods_validation = []
for model in models:
    log_likelihoods_model = []
    log_likelihoods_validation_model = []
    for i in range(20): #trimmed 
        validation_indices = split_indices[i].index
        training_indices = pd.concat( split_indices[0:i]+split_indices[i+1:] ).index
 
        log_likelihoods_model_i = []
        for u in range(len(users)):
            #FC the BIC or AIC criteria don't seem to prevent overfitting very well here,
            if model < 2: 
                mixture_model = sklearn.mixture.GMM(n_components=1+model, covariance_type='full') 
            else:
                mixture_model = sklearn.mixture.GMM(n_components=model-1, covariance_type='diag') 

            mixture_model.fit( np.array( users_feat_reduced.loc[training_indices].loc[users[u]] ) )
            bic = mixture_model.bic( np.array( users_feat_reduced.loc[training_indices].loc[users[u]] ) )
    
            log_likelihoods_model_i.append( pd.DataFrame( mixture_model.score( np.array(users_feat_reduced) ), index=users_feat_reduced.index, columns=[users[u]] ) )
            #log_likelihoods_model_i[model][-1].sort(ascending=True)
        #plot_hist_nn( np.histogram(log_likelihoods_model_i, np.linspace(np.min(log_likelihoods_model_i),np.max(log_likelihoods_model_i),num=100)) )

        log_likelihoods_model_i = log_likelihoods_model_i[0].join( log_likelihoods_model_i[1:] ) 
        
        log_likelihoods_model_i['user'] = log_likelihoods_model_i.index.get_level_values('user')
        log_likelihoods_model_i[['proba_user_%s'%user for user in users]] = log_likelihoods_model_i[list(users)].apply( exp_prop, axis=1 )
        log_likelihoods_model_i['pred'] = log_likelihoods_model_i[list(users)].apply(np.argmax, axis=1)
        log_likelihoods_model_i['accur'] = log_likelihoods_model_i.apply( lambda row:int(row['user'] == row['pred']), axis=1 )

        #print 'accuracy of model ', model ,'at i = ', i, ' is ', np.mean( log_likelihoods_model_i.loc[validation_indices]['accur'] )
        
        log_likelihoods_model.append( log_likelihoods_model_i )
        log_likelihoods_validation_model.append( log_likelihoods_model_i.loc[validation_indices] )
   
    log_likelihoods_model = pd.concat( log_likelihoods_model, keys=range(20), names=['i'] )
    log_likelihoods_validation_model = pd.concat( log_likelihoods_validation_model, keys=range(20), names=['i'] )

    log_likelihoods.append( log_likelihoods_model )
    log_likelihoods_validation.append( log_likelihoods_validation_model )

log_likelihoods = pd.concat( log_likelihoods, keys=models, names=['model'] )
log_likelihoods_validation = pd.concat( log_likelihoods_validation, keys=models, names=['model'] )

accuracies = log_likelihoods_validation.groupby(level=['model','i'])['accur'].apply(np.mean)
average_accuracies = accuracies.groupby(level=['model']).apply(np.mean)
std_accuracies = accuracies.groupby(level=['model']).apply(np.std)

print 'average_accuracies ', average_accuracies
print 'std_accuracies ', std_accuracies


###########################################################################


#NB 2 users, split of test data, use decision tree
newAverages=averages.reset_index(drop=False)

from sklearn import tree

#NB can change the user numbers, just to mess around
dataset1=newAverages[newAverages['user']==2] #<-------try other values
dataset2=newAverages[newAverages['user']==3] #<-------try other values
percentForTrain=.5                          #<-------try other values
features=['acce', 'acce_perp', 'acce_tang', 'speed'] #<-------try other values
#the full list is : ['acce', 'acce_perp', 'acce_tang', 'speed', 'sign_curv', 'trip_dura', 'trip_dist']

n=len(dataset1)
sepIndex=int(n*percentForTrain)

#the column, 'user' is the output Y
trainData=pd.concat([dataset1[0:sepIndex],dataset2[0:sepIndex]])
testData=pd.concat([dataset1[sepIndex:],dataset2[sepIndex:]])

#shuffle the rows of trainData, just because
trainData=trainData.reindex(np.random.permutation(trainData.index))

#make a decision Tree class:

T = tree.DecisionTreeClassifier()

#fit
T.fit(trainData.as_matrix(features),trainData.as_matrix(['user']))

#test the fit:
numCorrect=0
totalTests=len(testData)

for i in range(len(testData)):
    prediction=T.predict(testData[i:i+1].as_matrix(features))
    if prediction==testData['user'][i:i+1].as_matrix():
        numCorrect+=1
percentCorrect=float(numCorrect)/totalTests
print percentCorrect

#NB fewer features sometimes work even better?!

# <codecell>

testData[1:1+1]['user'].as_matrix()

