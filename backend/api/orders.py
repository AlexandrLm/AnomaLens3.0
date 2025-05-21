from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from typing import  Optional
from datetime import datetime

from .. import crud, schemas
from ..database import get_db

router = APIRouter()

@router.get("/", response_model=schemas.PaginatedResponse[schemas.Order])
async def read_orders(
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0, description="Количество записей для пропуска (пагинация)"),
    limit: int = Query(100, ge=1, le=1000, description="Максимальное количество записей для возврата"),
    start_date: Optional[datetime] = Query(None, description="Начальная дата (ISO формат) для фильтрации по дате покупки"),
    end_date: Optional[datetime] = Query(None, description="Конечная дата (ISO формат) для фильтрации по дате покупки")
):
    """
    Получает список заказов с детальной информацией.
    Поддерживает пагинацию (skip, limit) и фильтрацию по дате покупки.
    Предзагружает связанные данные: items, products, payments, reviews, customer.
    """
    orders = crud.get_orders_with_details(db, start_date=start_date, end_date=end_date, skip=skip, limit=limit)
    total_count = crud.get_order_count(db, start_date=start_date, end_date=end_date)
    
    items_response = [schemas.Order.model_validate(order) for order in orders]
    
    return schemas.PaginatedResponse(total=total_count, items=items_response)

@router.get("/{order_id}", response_model=schemas.Order)
async def read_order_by_id(order_id: str, db: Session = Depends(get_db)):
    """
    Получает информацию о конкретном заказе по его ID, 
    включая все связанные данные (товары, платежи, отзывы, покупателя).
    """
    db_order = crud.get_order_by_id_with_details(db, order_id=order_id)
    
    if db_order is None:
        raise HTTPException(status_code=404, detail="Order not found")
    
    return schemas.Order.model_validate(db_order)