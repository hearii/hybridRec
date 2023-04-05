from typing import Optional,List,Dict
from pydantic import BaseModel

class MovieRecommendation(BaseModel):
    movie_id:int
    movie_name: str
    user_id:str