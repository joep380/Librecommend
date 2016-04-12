# Librecommend
CSCE 470 Project: Reader-kickstart tool for people who are looking to get into literature.




To run this program:
 "python engine.py folder_path_to_books"
 
The results will be displayed where ever the code is being ran, but they are also written to 
the "output.txt" file (created in the current working directory) in the same format.

The user's query gets created as "query.txt" within the current working directory before the ranking 
and clustering algorithm are ran.

This program is optimized for the size of the small subset of books.
The subset of books is around 200 instead of the ~3000 from the entire set, this is just to 
cut down the runtime for testing purposes.

The results given are a list of top ranked books (using the cosine score ), and then 
groups (clusters) of books that are based on the given top ranked results (displayed in respective order).
With the first implementation we realized that doing clustering with cosine scores did not create 
useful clusters, and running the regular clustering algorithm with tf-idf scores was not very useful since the clusters 
would always be the same every time it was ran, so we found a way to correlate the cosine results to the centroids of 
the clusters.
