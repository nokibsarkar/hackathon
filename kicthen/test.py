from fastapi.testclient import TestClient
from src import app
from src.database import engine
from src.database.ingredients import Ingredient
from src.database import Session
import pytest
import json

client = TestClient(app)

@pytest.fixture
def db():
    with Session(engine) as session:
        yield session


def test_create_ingredient(db):
    response = client.post(
        "/ingredients/",
        json={"name": "sugar", "quantity": 5, "unit": "kg"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["name"] == "sugar"
    assert data["data"]["quantity"] == 5
    assert data["data"]["unit"] == "kg"
    assert data["success"] == True

def test_get_all_ingredients(db):
    response = client.get("/ingredients/")
    assert response.status_code == 200
    data = response.json()
    print(data)
    assert len(data["data"]) > 0
    assert data["success"] == True