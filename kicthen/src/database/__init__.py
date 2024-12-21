"""
generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model Ingredient {
  id        Int      @id @default(autoincrement())
  name      String
  quantity  Int
  unit      String
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt
}

"""
from fastapi import Depends
from sqlmodel import SQLModel, Field, create_engine, Session, select
from typing import Annotated, List, Generic, TypeVar
from .ingredients import *
from .recipe import *


DB_URL = 'sqlite:///./test.sqlite3'
engine = create_engine(DB_URL)

def get_db():
    db = Session(engine)
    try:
        yield db
    finally:
        db.close()

def init_db():
    # Create the database
    SQLModel.metadata.create_all(engine)
    ingredients = [
        Ingredient(name="oil", quantity=1, unit=Unit.ml),
        Ingredient(name="salt", quantity=1, unit=Unit.kg),
        Ingredient(name="beef", quantity=1, unit=Unit.kg),
        Ingredient(name="garlic", quantity=1, unit=Unit.piece),
    ]
    with Session(engine) as session:
        session.add_all(ingredients)
        session.commit()
T = TypeVar("T")

class ResponseSingle(SQLModel, Generic[T]):
    data: T | None
    message: str
    success: bool

class ResponseMultiple(SQLModel, Generic[T]):
    data: List[T]
    message: str
    success: bool



DATABASE = Annotated[Session, Depends(get_db)]