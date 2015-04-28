'''
Build a recommender system for the Yelp dataset.
* Content-based filtering
* User-user collaborative filtering

----------
AUTHOR: CHIA YING LEE
DATE: 10 APRIL 2015
----------
'''


import pandas as pd
import numpy as np
from numpy.linalg import norm
from sklearn.pipeline import Pipeline, FeatureUnion
from src import estimators, transformers
from scipy.sparse import coo_matrix
import heapq

# Load data
try:
    data_load_time
except NameError:
    execfile('src/load_data.py')
else:
    print 'Data was loaded at ' + data_load_time.time().isoformat()

# Personalized recommendation for a specific user
user = 'zzmRKNph-pBHDL2qwGv9Fw'

## ----------------
## CONTENT BASED FILTERING
## ----------------
print '*** Using Content-based Filtering for Recommendation ***'
print '** Initializing feature extraction for user ' + user

# Extract features of each business: category, attribute, average rating
OHE_cat = transformers.One_Hot_Encoder('categories', 'list', sparse=False)
OHE_attr= transformers.One_Hot_Encoder('attributes', 'dict', sparse=False)
OHE_city= transformers.One_Hot_Encoder('city', 'value', sparse=False)
rating = transformers.Column_Selector(['stars'])
OHE_union = FeatureUnion([ ('cat', OHE_cat), ('attr', OHE_attr), ('city', OHE_city), ('rating', rating) ])
OHE_union.fit(df_business)
print 'Done'

# Generate profile: weighted average of features for business she has reviewed
print '**Getting businesses...'
reviewed_businesses = df_review.ix[df_review.user_id == user]
reviewed_businesses['stars'] = reviewed_businesses['stars'] - float(df_user.average_stars[df_user.user_id == user])
idx_reviewed = [pd.Index(df_business.business_id).get_loc(b) for b in reviewed_businesses.business_id]

print '**Creating profile...'
features = OHE_union.transform(df_business.ix[idx_reviewed])
profile = np.matrix(reviewed_businesses.stars) * features
print 'Done'

# Given un-reviewed business, compute cosine similarity to user's profile
print '**Computing similarity to all businesses...'
idx_new = range(100) 
#[pd.Index(df_business.business_id).get_loc(b) for b in df_business.business_id if b not in reviewed_businesses.business_id]
features = OHE_union.transform(df_business.ix[idx_new])
similarity = np.asarray(profile * features.T) * 1./(norm(profile) * norm(features, axis = 1))
print 'Done'

# Output: recommend the most similar business
idx_recommendation = similarity.argmax()
print '\n**********'
print 'Hi ' + df_user.name[df_user.user_id == user].iget_value(0) + '!'
print 'We recommend you to visit ' + df_business.name[idx_recommendation] + ' located at '
print df_business.full_address[idx_recommendation]
print '**********'


## -------------------
## COLLABORATIVE FILTERING
## -------------------
print '*** Using Collaborative Filtering for Recommendation ***'

df_review['stars'] = df_review.groupby('business_id')['stars'].transform(lambda x : x - x.mean())

def get_idx(user_id): 
    global running_index
    running_index = running_index + 1
    return pd.Series(np.zeros(len(user_id)) + running_index) 
# For speed, get_idx assumes df_review and df_user contain the same users, and is fed in sorted order.
running_index = -1 
df_review['user_idx'] = df_review.groupby('user_id')['user_id'].transform(get_idx)

# Work in terms of sparse matrix
print '** Processing utility matrix...'

def convert_to_sparse(group):
    ratings = coo_matrix( (np.array(group['stars']), (np.array(group['user_idx']), np.zeros(len(group)))), 
                          shape = (len(df_user), 1) ).tocsc()
    return ratings / np.sqrt(float(ratings.T.dot(ratings).toarray()))

utility = df_review.groupby('business_id')[['stars', 'user_idx']].apply(convert_to_sparse) 

# Get top recommendatiokns
print '** Generating recommendations...'

def cosine_similarity(v1, v2):
    return float(v1.T.dot(v2).toarray())

def get_recommended_businesses(n, business_id):
    util_to_match = utility[utility.index == business_id]
    similarity = utility.apply(lambda x: cosine_similarity(util_to_match.values[0], x))
    similarity.sort(ascending=False)
    return similarity[1:(n+1)]

fav_business = df_review.business_id[ df_review.stars[ df_review.user_id == user ].argmax() ]

rec = pd.DataFrame(get_recommended_businesses(5, fav_business), columns=['similarity'])
rec['name'] = [ df_business.name[ df_business.business_id == business_id ].values[0] for business_id in rec.index]
print 'Done'

# Output recommendation
print 'Hi ' + df_user.name[df_user.user_id == user].values[0] + '!\nCheck out these businesses!'
for name in rec.name:
    print name
