'''
Script to load data from file. The exact columns to load should
be specified. 

----------
AUTHOR: CHIA YING LEE
DATE: 10 APRIL 2015
----------
'''

import pandas as pd
import simplejson as json
from datetime import datetime
from sklearn.cross_validation import train_test_split

print '**Loading data...'

# LOAD DATA FOR TYPE = dataset_type
fileheading = 'data/raw/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_'

def get_data(line, cols):
    d = json.loads(line)
    return dict((key, d[key]) for key in cols)

# Load business data
cols = ('business_id', 'name')
with open(fileheading + 'business.json') as f:
    df_business = pd.DataFrame(get_data(line, cols) for line in f)
df_business = df_business.sort('business_id')
df_business.index = range(len(df_business))

# Load user data
cols = ('user_id', 'name')
with open(fileheading + 'user.json') as f:
    df_user = pd.DataFrame(get_data(line, cols) for line in f)
df_user = df_user.sort('user_id')
df_user.index = range(len(df_user))

# Load review data
cols = ('user_id', 'business_id', 'stars')
with open(fileheading + 'review.json') as f:
    df_review = pd.DataFrame(get_data(line, cols) for line in f)

data_load_time = datetime.now()
print 'Data was loaded at ' + data_load_time.time().isoformat()
