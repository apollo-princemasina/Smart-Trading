from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class PredictionRecord(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    prediction = Column(String)
