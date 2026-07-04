from fastapi import APIRouter
router = APIRouter(prefix='/signals')

@router.get('/')
def get_signals():
    return {'signals': []}
