from fastapi import FastAPI
import uvicorn
from router import recommendation_route

app=FastAPI()
app.include_router(recommendation_route)


