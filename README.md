# Objective

The objective of this project is to use the [Yelp dataset](http://www.yelp.com/dataset_challenge) for two problems:

1. predict ratings of newly added businesses;
2. build a simple recommender system to recommend new businesses to users.

The dataset contains data on businesses, reviews and users.

## Predict ratings of new businesses

We use the following predictors in `src/analysis.py`

* Location-based model: predict based on ratings of nearest neighbors (using `sklean.neighbors.KNeighborsRegressor`)
* Categories and attributes model: predict based on the similarity of a business' category and attributes to those of existing businesses, using one-hot encoding `src.transformer.One_Hot_Encoder` (see [Custom transformer classes](./README.md#Transformers) below)

## Recommend new businesses to users

We use the following recommenders in `src/recommender.py`

* Content-based filtering
* Collaborative filtering

### Content-based filtering

Recommend new businesses based on the similarity of a business to a user's profile. Businesses are characterized by a feature vector comprising of a `FeatureUnion` of

* Average review score
* Location (one-hot encoding of city using `One_Hot_Encoder`)
* Business categories and attributes (using `One_Hot_Encoder`)

A user's profile is a weighted average of the features of the businesses she have reviewed, weighted by her rating of the business.


### Collaborative filtering

Recommend new businesses that appeal to similar users, based on utility matrix of each user's rating of each business, mean-normalized by businesses' average rating. (The utility matrix is represented as a `pandas.Series` of `scipy.sparse` vectors of user ratings for each business.)
For each business, the vector of user ratings represents the business' appeal.
The businesses whose appeal are closest (by cosine similarity) to the user's favorite business are recommended to the user.


## Custom Classes

### Transformers

The data can be transformed using transformers in the module `src/transformers.py`.

* `Column_Select_Transformer(colnames)` selects the specified column(s) given in the list `colnames`.
* `One_Hot_Encoder(colnames, value_type = 'list', sparse = True)` turns the specified column in `colnames` with values of `value_type` into a `sparse` one-hot encoding. `value_type` must either be `list` or `dict`. The features of the transformed matrix are all distinct entries or keys in `colnames`.

