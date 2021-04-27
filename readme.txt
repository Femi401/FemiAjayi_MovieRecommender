movieRecom1.py
to run the codes you need to have two csv files. movies.csv and ratings.csv in the default path.
the code should be run part by part. different parts is separated clearly in the codes.
you may recieve an error while running this function recommendedMoviesAsperItemSimilarity(user_id) for some users.
there is a conditin in this function: the user similarity score should be more than 0.4 out of one. (sorted_movies_as_per_userChoice=sorted_movies_as_per_userChoice[sorted_movies_as_per_userChoice['similarity'] >=0.4]['movie_id'])
if this condition not satisfied you will recieve and error which is very rare.

for the comparison part which is at the end of the code, you need one more csv file: recc_with_ids.csv
all files in the submitted RAR file.


For recmovie2
-run it using google colab
-requires 2 files from brightspace, movies and ratings to be uploaded
-run segment by segment
-if error occurs please reset runtime and try again or try running the previous block again