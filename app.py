from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from functools import lru_cache
from sklearn.feature_extraction.text import CountVectorizer

# deploy locally uvicorn app:app --reload

app = FastAPI()

movies_file_path = "files/df_clean.csv"

@lru_cache()
def get_movies_df(file_path):
    movies_df = pd.read_csv(file_path)
    return movies_df

movies_df = get_movies_df(movies_file_path)

class Movies(BaseModel):
    titles: str


@app.get("/")
async def root():
    return {"Movie recommender alive and kicking!"}


@app.post("/movies/")
async def create_item(movies: Movies):
    movies_titles = movies.titles
    return get_recommendations_list(movies_titles)


# Function that takes in a list of movie titles as input and outputs most similar movies ON THE FLY
def get_recommendations_list(movie_titles, similar_movies=10):
    movies_list = movie_titles.split(", ")

    vectorized_count_matrix = get_vector_count_matrix()
    # Get the index of the movies that matches the title
    movie_id_list = []
    for movie_title in movies_list:
        movie_id = movies_df.index[movies_df["primaryTitle"] == movie_title]
        movie_id_list.append(movie_id[0])

    df_sim_list = pd.DataFrame()
    print(movie_id_list)
    for movie_id in movie_id_list:
        vector = vectorized_count_matrix[movie_id]
        simil = vector.dot(vectorized_count_matrix.T).todense()[0]
        sim_scores = np.squeeze(np.array(simil))
        df_sim = pd.DataFrame(sim_scores.reshape(-1), columns=[movie_id])
        df_sim_list[movie_id] = df_sim

    df_sim_list["average"] = df_sim_list.mean(numeric_only=True, axis=1)

    rslt_df = df_sim_list.sort_values(by="average", ascending=False)

    rec_list = rslt_df.reset_index()
    list1 = []
    list1 = rec_list["index"][
        len(movie_titles) : (len(movie_titles) + similar_movies)
    ].values

    # Return the top 10 most similar movies
    return movies_df[["primaryTitle", "genres"]].iloc[list1]


def get_vector_count_matrix():
    vector_count = CountVectorizer(stop_words="english")
    return vector_count.fit_transform(movies_df["soup"])
