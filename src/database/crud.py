from .models import PredictionRecord

def save_prediction(session, record):
    session.add(record)
    session.commit()
