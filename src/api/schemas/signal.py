from pydantic import BaseModel

class Signal(BaseModel):
    symbol: str
    direction: str
    confidence: float
