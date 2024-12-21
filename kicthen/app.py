from dotenv import load_dotenv

load_dotenv()
from src import app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)