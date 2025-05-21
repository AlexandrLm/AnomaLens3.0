from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional

from .. import crud, schemas
from ..database import get_db

router = APIRouter()

@router.post("/", response_model=schemas.Setting, status_code=status.HTTP_201_CREATED)
async def create_setting_endpoint(
    setting: schemas.SettingCreate, db: Session = Depends(get_db)
):
    """Создает новую настройку."""
    db_setting = crud.get_setting_by_key(db, key=setting.key)
    if db_setting:
        raise HTTPException(status_code=400, detail="Setting key already exists")
    return crud.create_setting(db=db, setting=setting)

@router.get("/", response_model=List[schemas.Setting])
async def read_settings_endpoint(
    skip: int = 0, limit: int = 100, db: Session = Depends(get_db)
):
    """Получает список всех настроек."""
    settings = crud.get_settings(db, skip=skip, limit=limit)
    return [schemas.Setting.model_validate(s) for s in settings]

@router.get("/{key}", response_model=schemas.Setting)
async def read_setting_by_key_endpoint(key: str, db: Session = Depends(get_db)):
    """Получает настройку по ключу."""
    db_setting = crud.get_setting_by_key(db, key=key)
    if db_setting is None:
        raise HTTPException(status_code=404, detail="Setting not found")
    return schemas.Setting.model_validate(db_setting)

@router.put("/{key}", response_model=schemas.Setting)
async def update_setting_endpoint(
    key: str, setting_update: schemas.SettingUpdate, db: Session = Depends(get_db)
):
    """Обновляет значение настройки по ключу."""
    db_setting = crud.update_setting(db, key=key, setting_update=setting_update)
    if db_setting is None:
        raise HTTPException(status_code=404, detail="Setting not found")
    return schemas.Setting.model_validate(db_setting)

@router.delete("/{key}", response_model=schemas.Setting)
async def delete_setting_endpoint(key: str, db: Session = Depends(get_db)):
    """Удаляет настройку по ключу."""
    db_setting = crud.delete_setting(db, key=key)
    if db_setting is None:
        raise HTTPException(status_code=404, detail="Setting not found")
    return schemas.Setting.model_validate(db_setting) 