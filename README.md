# IMDB-Score-Prediction
This model predicts the IMDB score using neural network.

Test set loss: 0.077827</br>
Test set accuracy: 0.725404</br>
Train set loss: 0.060578</br>
Train set accuracy: 0.797357

## Data Description
The dataset namely, IMDB 5000 MOVIE DATASET is taken from Kaggle repository. It contains 28 variables for 5043 movies. The attributes are:
* Color (black n white or colored)
*	director_name
* num_critic_for_reviews
* duration
* director_facebook_likes
* actor_3_facebook_likes
* actor_2_name
* actor_1_facebook_likes
* gross
* genres
* actor 1 name
* movie title
* num_voted_users
* cast_total_facebook_likes
* actor_3_name
* facenumber_in_poster
* plot_keywords
* movie_imdb_link
* num_user_for_reviews
* language
* country
* content_rating
* budget
* title_year
* actor_2_facebook_likes
* imdb_score
* aspect_ratio
* movie_facebook_likes

Target variable : imdb_score

[Dataset Source](https://www.kaggle.com/carolzhangdc/imdb-5000-movie-dataset)

## Dependencies: ##
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* keras
