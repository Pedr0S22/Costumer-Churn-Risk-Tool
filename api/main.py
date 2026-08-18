from fastapi import FastAPI
from api.routers import scoring
from api.services.scoring_service import scoring_service

app = FastAPI(title="Costumer Churn Risk API")

# Include the MVC routers
app.include_router(scoring.router)

@app.on_event("startup")
def startup_event():
    # Load the machine learning model on application startup
    scoring_service.load_model()
