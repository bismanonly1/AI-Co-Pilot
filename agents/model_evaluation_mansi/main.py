from fastapi import FastAPI
from agents import evaluation_api
app = FastAPI()
app.include_router(evaluation_api.router)
