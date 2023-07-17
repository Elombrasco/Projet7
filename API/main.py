from fastapi import FastAPI
from pydantic import BaseModel
from model import predict

app = FastAPI()

class id_in(BaseModel):
	customer : int

class score_out(id_in):
	prediction: dict


@app.get('/')
async def root():
	return {"message" : "Welcome to Score Prediction API"}

@app.post("/predict", response_model=score_out, status_code=200)
def get_prediction(payload: id_in):
	customer = payload.customer
	predicted_score = predict(customer)
	response_object = {"customer_id":customer, "score": predicted_score}

	return response_object

if __name__ == "__main__":
	uvicorn.run(app)