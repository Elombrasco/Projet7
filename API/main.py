from fastapi import FastAPI
import pickle
from lightgbm import LGBMClassifier
import pandas as pd

model = pickle.load(open("LGBMClassifier.pkl", "rb"))

app = FastAPI()

@app.get('/'):
async def index():
	return {"Message" : "Welcome to Score Prediction API"}