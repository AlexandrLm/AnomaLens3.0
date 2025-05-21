from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List

from .. import crud, schemas, models
from ..database import get_db

router = APIRouter()

@router.get("/", response_model=schemas.PaginatedResponse[schemas.Geolocation])
async def read_geolocation_entries(
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0, description="Количество записей для пропуска"),
    limit: int = Query(100, ge=1, le=1000, description="Максимальное количество записей")
):
    """Получает список записей геолокации с пагинацией."""
    geo_entries = crud.get_items(db, model=models.Geolocation, skip=skip, limit=limit)
    total_count = crud.get_item_count(db, model=models.Geolocation)
    items_response = [schemas.Geolocation.model_validate(entry) for entry in geo_entries]
    return schemas.PaginatedResponse(total=total_count, items=items_response)
