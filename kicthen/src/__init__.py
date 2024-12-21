from fastapi import FastAPI


from .routes import ingredients_router, chat_router
from .database import init_db
class KitchenManager(FastAPI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_router(ingredients_router)
        self.include_router(chat_router)
        # init_db()
app = KitchenManager(
    title="Kitchen Manager",
    description="Manage your kitchen inventory",
    version="0.1.0",
    responses = {
        404: {"data": None, "message": "Not found", "success": False}},

)