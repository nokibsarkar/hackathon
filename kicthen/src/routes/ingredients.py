from fastapi import APIRouter
from typing import Generic, TypeVar, Any, List
from sqlmodel import select
from sqlmodel import SQLModel
from ..database import DATABASE, IngredientCreate, Ingredient, IngredientUpdate, ResponseSingle, ResponseMultiple
ingredients_router = APIRouter(
    prefix="/ingredients",
    tags=["ingredients"],
    
)


@ingredients_router.post('/', response_model=ResponseSingle[Ingredient])
def create_ingredent(conn : DATABASE, ingredient: IngredientCreate):
    """
    Create a new ingredient
    """
    try:
        quantity = ingredient.quantity
        unit = ingredient.unit
        name = ingredient.name
        new_ing = Ingredient(name=name, quantity=quantity, unit=unit)
        conn.add(new_ing)
        conn.commit()
        conn.refresh(new_ing)
        resp = ResponseSingle[Ingredient](data=new_ing, message="Ingredient created successfully", success=True)
        return resp
    except Exception as e:
        conn.rollback()
        return ResponseSingle[Ingredient](data=None, message="An error occurred", success=False)




@ingredients_router.get('/', response_model=ResponseMultiple[Ingredient])
def get_all_ingredient(conn : DATABASE):
  """
  Get all ingredients
  """
  try:
    stmt = select(Ingredient).order_by(Ingredient.name)
    ingredients = conn.exec(stmt).all()
    resp = ResponseMultiple[Ingredient](data=ingredients, message="Ingredients retrieved successfully", success=True)
    return resp
  except Exception as e:
    return ResponseMultiple[Ingredient](data=[], message="An error occurred", success=False)




@ingredients_router.get("/{id}", response_model=ResponseSingle[Ingredient])
def get_ingredient(id : str, conn : DATABASE):
    """
        get a particular ingredient
    """
    try:
        stmt = select(Ingredient).where(Ingredient.name == id).limit(1)
        ingredient = conn.exec(stmt).first()
        if not ingredient:
            raise Exception("Ingredient not found")
        return ResponseSingle[Ingredient](data=ingredient, message="Ingredient retrieved successfully", success=True)
    except Exception as e:
        return ResponseSingle[Ingredient](data=None, message="An error occurred", success=False)





@ingredients_router.patch("/{id}", response_model=ResponseSingle[Ingredient])
def update_ingredient(id : str, conn : DATABASE, update : IngredientUpdate):
    """
    Update a particular ingredient
    """
    try:
        stmt = select(Ingredient).where(Ingredient.name == id).limit(1)
        ingredient = conn.exec(stmt).first()
        if not ingredient:
            raise Exception("Ingredient not found")
        # update the ingredient
        if update.name:
            ingredient.name = update.name
        if update.quantity:
            ingredient.quantity = update.quantity
        if update.unit:
            ingredient.unit = update.unit
        conn.commit()
        conn.refresh(ingredient)
        return ResponseSingle[Ingredient](data=ingredient, message="Ingredient updated successfully", success=True)
    except Exception as e:
        conn.rollback()
        return ResponseSingle[Ingredient](data=None, message="An error occurred", success=False)





@ingredients_router.delete("/{id}", response_model=ResponseSingle[Ingredient])
def delete_ingredient(id : int, conn : DATABASE):
    """
    Delete a particular ingredient
    """
    try:
        stmt = select(Ingredient).where(Ingredient.id == id).limit(1)
        ingredient = conn.exec(stmt).first()
        if not ingredient:
            raise Exception("Ingredient not found")
        conn.delete(ingredient)
        conn.commit()
        return ResponseSingle[Ingredient](data=ingredient, message="Ingredient deleted successfully", success=True)
    except Exception as e:
        conn.rollback()
        return ResponseSingle[Ingredient](data=None, message="An error occurred", success=False)