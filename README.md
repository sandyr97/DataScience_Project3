# DataScience_Project3
Weekend Movie Trip: Use clustering models to determine similar movies to recommend using the other ratings and tags of movies by other users as features

Steps:
1.	Load data into data frame
a.	Movie
b.	Links
c.	Ratings
2.	Clean data
a.	Check for NAN values
3.	Feature Engineering
a.	genres of movie
b.	years
4.	Visualization
a.	Seaborne heat map for correlation
5.	KMeans
6.	MeanShift
I used visualization to find the features with the highest correlation to cluster. 
These features were reviews and tags which I clustered based on the genre of the movie. 
I used clustering methods KMeans and Meanshift to classify these attributes. My KMeans 
cluster was not very accurate but my meanShift cluster was better and created 5 distinct clusters for tags vs reviews.
