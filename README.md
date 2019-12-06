# DataScience_Project3
Weekend Movie Trip: Use clustering models to determine similar movies to recommend using the other ratings and tags of movies by other users as features

Steps:
1.	Load data into data frame <br />
a.	Movie <br />
b.	Links <br />
c.	Ratings <br />
d.  Tags <br />
2.	Clean data <br />
a.	Check for NAN values <br />
3.	Feature Engineering <br />
a.	genres of movie <br />
b.	years <br />
4.	Visualization <br />
a.	Seaborne heat map for correlation <br />
5.	KMeans
6.	MeanShift
I used visualization to find the features with the highest correlation to cluster. 
These features were reviews and tags which I clustered based on the genre of the movie. 
I used clustering methods KMeans and Meanshift to classify these attributes. My KMeans 
cluster was not very accurate but my meanShift cluster was better and created 5 distinct clusters for tags vs reviews.
