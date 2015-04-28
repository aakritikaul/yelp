# Objective

The [Yelp dataset](http://www.yelp.com/dataset_challenge) contains data on businesses, users, as well as user reviews of businesses. This project builds a simple recommender system for recommending new businesses to users, using data on user ratings and business location, and other characteristics.

In `src/recommender.py`, we implement two types of recommender systems

* Content-based filtering
* Collaborative filtering



### Content-based filtering

Content-based filtering recommends new businesses based on the similarity of a business' characteristics to a user's profile. Each business is characterized by its

* Average review score
* Location (city)
* Category (e.g. restaurant, dentist)
* Attributes (e.g. non-smoking, accepts credit card)

The city, category and attribute features were represented by one-hot encoding using a custom class `One_Hot_Encoder` in `src/transformers.py`, thereafter the feature vector was formed using `FeatureUnion` in the `sklearn.pipeline` module.

A user's profile is a weighted average of the features of the businesses she reviewed, weighted by her rating of the business.
Finally, the algorithm recommends the business(es) that are closest in cosine distance to the user's profile.


### Collaborative filtering

Collaborative filtering recommends new businesses that appeal to similar users. Each user's rating of each business is recorded as an entry in a utility matrix.
For each business, the vector of user ratings represents how much the business appeals to subgroups of users.
The algorithm recommends the businesses whose vector of user ratings are closest (in cosine similarity) to that of the user's favorite business.

The utility matrix is represented as a `pandas.Series` of `scipy.sparse` vectors of user ratings for each business. Ratings are centered to zero so that each businesses' average rating is zero, and missing values are set to zero.





## Custom Classes

The module `src/transformers.py` contains two transformer for transforming data.

* `Column_Selector(colnames)` selects the specified column(s) given in the list `colnames`.
* `One_Hot_Encoder(colnames, value_type = 'list', sparse = True)` turns the specified column in `colnames` with values of `value_type` into a `sparse` one-hot encoding. `value_type` must either be `list` or `dict`. The features of the transformed matrix are all distinct entries or keys in `colnames`. <br />
Dependencies: `sklearn.feature_extraction.DictVectorizer`.

