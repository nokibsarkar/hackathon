from sqlmodel import SQLModel, Field


class Recipe(SQLModel, table=True):
    recipe_id: int | None = Field(default=None, primary_key=True)
    