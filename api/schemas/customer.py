from pydantic import BaseModel
from typing import Optional, List

class CustomerRecord(BaseModel):
    customer_id: str
    signup_date: str
    last_login_date: Optional[str] = None
    plan_type: str
    monthly_spend: float
    num_logins_30d: int
    support_tickets_30d: int
    preferred_language: str
    
class RiskScoreResponse(BaseModel):
    customer_id: str
    risk_score: int
    risk_category: str
    top_reasons: List[str]
