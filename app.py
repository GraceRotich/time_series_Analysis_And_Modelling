from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import pickle

app = FastAPI()

class PredictRequest(BaseModel):
    #User inputs the data in string format
    date: str 
        
# Load the saved SARIMA model
with open('sarima_model.pkl', 'rb') as f:
    sarima_model = pickle.load(f)

@app.get("/")
def read_root():
    return {"message": "House Value Forecasting using SARIMA MODEL API"}

@app.post("/predict")
def predict(request: PredictRequest):
    # Parse the input date
    target_date = datetime.strptime(request.date, '%Y-%m-%d')
    
    # Calculate the number of months to forecast
    current_date = sarima_model.data.dates[-1]  
    current_date = datetime.strptime(str(current_date), '%Y-%m-%d %H:%M:%S')
    
    if target_date <= current_date:
        return {"error": "The target date must be after the last date in the training data."}
    
    # Calculate the number of months between the dates
    steps = (target_date.year - current_date.year) * 12 + target_date.month - current_date.month
    
    # Make predictions using the SARIMA model
    forecast = sarima_model.forecast(steps=steps)
    
    # Convert the forecast to a list
    forecast_list = forecast.tolist()
    
    return {"forecast": forecast_list}
