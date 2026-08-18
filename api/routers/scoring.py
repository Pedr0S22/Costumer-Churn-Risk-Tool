from fastapi import APIRouter, HTTPException
from api.schemas.customer import CustomerRecord, RiskScoreResponse
from api.services.scoring_service import scoring_service

router = APIRouter()

@router.post("/costume-record", response_model=RiskScoreResponse)
def score_customer_endpoint(record: CustomerRecord):
    try:
        result = scoring_service.score_customer(record)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=503, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
