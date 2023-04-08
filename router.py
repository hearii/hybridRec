from fastapi import APIRouter
from model import MovieRecommendation
from algo import hybrid
recommendation_route=APIRouter(prefix='/recommendation',tags=['Recommendations'])

@recommendation_route.post("/")
async def recommendations_movies(data :MovieRecommendation ):
    return hybrid(data.user_id,data.movie_name)

