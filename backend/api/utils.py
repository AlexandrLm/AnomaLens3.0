from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from typing import Type, TypeVar
from pydantic import BaseModel

from .. import crud, schemas
from ..database import Base, get_db

ModelType = TypeVar("ModelType", bound=Base)
SchemaType = TypeVar("SchemaType", bound=BaseModel)

def create_generic_router(
    model: Type[ModelType],
    response_schema: Type[SchemaType],
    prefix: str,
    tags: list[str],
    get_all_route: bool = True,
    get_one_route: bool = True
) -> APIRouter:
    """Создает APIRouter с базовыми CRUD эндпоинтами (GET all, GET one by ID)."""
    
    router = APIRouter(prefix=prefix, tags=tags)
    primary_key_name = model.__mapper__.primary_key[0].name
    if not hasattr(response_schema, 'model_validate'):
        raise AttributeError(f"Schema {response_schema.__name__} must have a model_validate method.")

    if get_all_route:
        @router.get("/", response_model=schemas.PaginatedResponse[response_schema])
        async def read_items(
            db: Session = Depends(get_db),
            skip: int = Query(0, ge=0, description="Количество записей для пропуска"),
            limit: int = Query(100, ge=1, le=1000, description="Максимальное количество записей")
        ):
            items_db = crud.get_items(db, model=model, skip=skip, limit=limit)
            total_count = crud.get_item_count(db, model=model)
            items_response = [response_schema.model_validate(item) for item in items_db]
            return {"total": total_count, "items": items_response}

    if get_one_route:
        @router.get(f"/{{{primary_key_name}}}", response_model=response_schema)
        async def read_item_by_id(item_id: str, db: Session = Depends(get_db)):
            db_item = crud.get_item_by_id(db, model=model, item_id=item_id)
            if db_item is None:
                raise HTTPException(status_code=404, detail=f"{model.__name__} not found")
            return response_schema.model_validate(db_item)
            
    return router 