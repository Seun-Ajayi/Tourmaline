import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from fastapi.responses import JSONResponse

from elbaite.ml.data import process_data
from elbaite.ml.model import inference
from elbaite.train import cat_features
from elbaite.utils import load_asset


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()

def hyphen_to_underscore(fieldname):
    return str(fieldname).replace("_", "-")

class Input(BaseModel):
    age: int = Field(..., example=39)              
    workclass: str = Field(..., example="State-gov")          
    fnlgt: int = Field(..., example=77516)              
    education: str = Field(..., example="Bachelors")          
    education_num: int = Field(..., example=13)      
    marital_status: str = Field(..., example="Never-married")     
    occupation: str = Field(..., example="Adm-clerical")         
    relationship: str= Field(..., example="Not-in-family")       
    race: str = Field(..., example="White")               
    sex: str = Field(..., example="Male")                
    capital_gain: int = Field(..., example=2174)      
    capital_loss: int = Field(..., example=0)       
    hours_per_week: int = Field(..., example=40)     
    native_country: str = Field(..., example="United-States") 
    


    class Config:
        alias_generator = hyphen_to_underscore
        allow_population_by_field_name = True


@app.get("/")
async def root():
    return JSONResponse(
        status_code=200,
        content={"Message": "Welcome to Tourmaline App! It may seem like you are interested in making some predictions..."},
    )

@app.post("/model/")
async def predict(data: Input):
    model = load_asset("trained_model.pkl")
    encoder = load_asset("encoder.pkl")
    lb = load_asset("lb.pkl")

    df = pd.DataFrame(data.dict(by_alias=True), index=[0])

    X, *_ = process_data(
        df, 
        categorical_features=cat_features, 
        label=None , 
        encoder=encoder,
        training=False, 
        lb=lb)

    predictions = inference(model, X)
    return JSONResponse(status_code=200, content=predictions.tolist())
