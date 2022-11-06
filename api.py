from fastapi import FastAPI
from starlette.requests import Request
from fastapi.templating import Jinja2Templates
from ml_model import Model, tokenizer_fucntion
import logging
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")
log = logging.getLogger('uvicorn')
tokenizer = tokenizer_fucntion

@app.get("/")
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request:Request):
    if request.method == "POST":
        form = await request.form()
        if form["message"]:
            data = form["message"]
            model = Model()
            return templates.TemplateResponse("index.html",  {"request": request, "prediction": model.predict(data), "og_message": form["message"]})
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)