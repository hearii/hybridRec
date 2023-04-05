from fastapi import FastAPI
import uvicorn
from router import recommendation_route

app=FastAPI()
app.include_router(recommendation_route)



if __name__=='__main__':
    uvicorn.run("app:app", host="127.0.0.1", port=5000, log_level="info", reload=True)
