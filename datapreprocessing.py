import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv(r"...\movie_metadata.csv")

sns.heatmap(df.isnull(), cbar= False)

df.describe(include = [np.number])
df.describe(include = ['O'])

duplicates = df.duplicated(subset='movie_title', keep=False)
sum(duplicates)

#Remove duplicates,keep only first
data=df.drop_duplicates(subset='movie_title',keep='first')
nandata=data.isnull().sum().to_frame('Number of NaN')

median=data[['gross','budget']].median()
data=data.fillna(median)

data = data.drop(['color','director_name','movie_imdb_link','aspect_ratio','plot_keywords','facenumber_in_poster','title_year',
                  'num_critic_for_reviews','cast_total_facebook_likes','language','country','budget','movie_title'], axis=1)
sns.heatmap(data.isnull(), cbar= False)
data = data.dropna()

cor=data.corr(method='pearson')
ax = sns.heatmap(cor, xticklabels=cor.columns, yticklabels=cor.columns, linewidths=.2, cmap="YlGnBu")

data.corr()['imdb_score'].sort_values(ascending = False).plot(kind='bar')

data['content_rating'].value_counts().plot(kind='bar')
data.groupby('content_rating').imdb_score.max().plot(kind='bar')

labels = [3,7,9,11]

from sklearn.preprocessing import LabelEncoder
for i in labels:
    data.iloc[:,i] = LabelEncoder().fit_transform(data.iloc[:,i])

clean = data.genres.str.split('|', expand=True).stack()
clean_df = pd.get_dummies(clean).groupby(level=0).sum()

data['imdb_score'] = pd.cut(data.imdb_score,[0,2,4,6,8,10], right=False)
imdb_rating = data['imdb_score']
data = data.drop(['imdb_score','genres'], axis=1)

newdata = pd.concat([data,clean_df,imdb_rating], axis=1)

newdata = data.to_csv(r"...\data.csv",index=False)
