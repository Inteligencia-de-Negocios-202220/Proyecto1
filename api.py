from fastapi import FastAPI
from ml_model import Model, tokenizer_fucntion
import logging
import uvicorn


app = FastAPI()
log = logging.getLogger('uvicorn')
tokenizer = tokenizer_fucntion

@app.get("/")
def root():
    return { "test": "test" }

@app.post("/predict")
def predict(data:str):
    model = Model()
    return {"prediction": model.predict(data)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)