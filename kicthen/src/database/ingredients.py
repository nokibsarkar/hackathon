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
from sqlmodel import SQLModel, Field
from enum import Enum
class Unit(Enum):
    kg = "kg"
    g = "g"
    l = "l"
    ml = "ml"
    cup = "cup"
    tbsp = "tbsp"
    piece = "piece"



class Ingredient(SQLModel, table=True):
    name: str = Field(primary_key=True)
    quantity: int
    unit: Unit
    # createdAt: datetime = Field(default=None)
    # updatedAt: datetime = Field(default=None)
    # createdAt: datetime = Field(default=None)
    # updatedAt: datetime = Field(default=None)
    # createdAt: datetime = Field(default=None)

class IngredientCreate(SQLModel):
    name: str
    quantity: int
    unit: Unit
class IngredientUpdate(SQLModel):
    name: str | None = None
    quantity: int | None = None
    unit: Unit | None = None