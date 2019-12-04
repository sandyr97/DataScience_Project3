import pandas as pd
import numpy as np

df_links=pd.read_csv("../data/links.csv")
df_movies=pd.read_csv("../data/movies.csv")
df_ratings=pd.read_csv("../data/ratings.csv")
df_tags=pd.read_csv("../data/tags.csv")

df_links.head(10)

df_movies

df_ratings.head(10)

df_tags.head(10)


print(df_movies.isna().any())
print(df_links.isna().any())
print(df_tags.isna().any())

df_links.dropna(inplace=True)
print(df_movies.isna().any())
print(df_links.isna().any())
print(df_tags.isna().any())

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# # Feature Engineering

# Created new columns in movie dataframe to define genres of that movie


gList=['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror','IMAX', 'Musical','Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
for i in gList:
    df_movies[i]=0

df_movies.head(10)



for i,row in df_movies.iterrows():
    genres=str(row['genres'])
    if genres!=('no genres listed'):
        genres=genres.split('|')
        for g in genres:
            df_movies.at[i,g]=1
df_movies.head(10)


for i,row in df_movies.iterrows():
    wholeTitle=str(row['title'])
    first=wholeTitle.find("(1")
    second=wholeTitle.find("(2")
    if(first!=-1):
        year=wholeTitle[first+1:first+5]
    elif(second!=-1):
        year=wholeTitle[second+1:second+5]
    else:
        year=0
    df_movies.at[i,'year']=int(year)
df_movies.head(10)

c=0
for i,row in df_movies.iterrows():
    if c>3:
        break
    mid=row['movieId']
    rows = df_ratings.loc[df_ratings['movieId']==mid]['rating']
    row2=df_tags.loc[df_tags['movieId']==mid]['tag']
    rating=rows.mean()
    reviews=rows.count()
    tags=row2.count()
    if rating != np.nan:
        df_movies.at[i,'Rating']=rating
    if reviews != np.nan:
            df_movies.at[i,'Reviews']=reviews
    if tags != np.nan:
            df_movies.at[i,'Tags']=tags

df_movies.head(10)


# # Visualization

import seaborn as sns
fig, ax = plt.subplots(figsize=(10,10))
cl = df_movies[['year','Rating','Reviews', 'Tags']].corr()
sns.heatmap(cl, square = True, ax=ax)


# Although correlations aren't too great, reviews and tags have the best correlation, I am going to cluster with those two features. I am also going to cluster rating and reviews since it has the second best correlation.

'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror','IMAX', 'Musical','Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'


# # KMeans Clustering

X=df_movies[['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror','IMAX', 'Musical','Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']]
km=KMeans(7)
km.fit(X)

prediction=km.predict(X)
centriods=km.cluster_centers_
kmeans=pd.DataFrame(prediction)

fig=plt.figure()
ax=fig.add_subplot(111)
scatter=ax.scatter(df_movies['Reviews'], df_movies['Tags'],c=kmeans[0],s=50)
ax.set_title('Clustering with K-Means')
ax.set_xlabel('Reviews')
ax.set_ylabel('Tags')
plt.xlim(0, 100)
plt.ylim(0, 10)
plt.colorbar(scatter)
plt.show()


# Clusters are not very clear

fig=plt.figure()
ax=fig.add_subplot(111)
scatter=ax.scatter(df_movies['Rating'], df_movies['Reviews'],c=kmeans[0],s=50)
ax.set_title('Clustering with K-Means')
ax.set_xlabel('Rating')
ax.set_ylabel('Reviews')
plt.ylim(0, 200)
plt.xlim(2, 4.5)
plt.colorbar(scatter)
plt.show()


# Clusters still aren't clear, so I am going to try a different clustering method

# # Meanshift Clustering

from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs

list1=list(df_movies['Reviews'])
list2=list(df_movies['Tags'])
list3= [list(a) for a in zip(list1, list2)]
clusters=np.array(list3)
clusters


X, _ = make_blobs(n_samples = 150, centers = clusters, cluster_std = 0.60)
ms = MeanShift()
ms.fit(X)
cluster_centers = ms.cluster_centers_

fig = plt.figure()

ax = fig.add_subplot(111)

ax.scatter(X[:, 0], X[:, 1], marker ='o')

ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker ='x', color ='red', s = 300, linewidth = 5, zorder = 10)
ax.set_xlabel('Reviews')
ax.set_ylabel('Tags')
plt.show()


# Clusters are clearer than KMeans method