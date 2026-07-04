from fastapi import APIRouter
router = APIRouter(prefix='/predictions')

@router.get('/')
def get_predictions():
    return {'predictions': []}
