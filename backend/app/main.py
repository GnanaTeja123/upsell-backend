from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.endpoints import router as api_router

app = FastAPI(
    title="Customer Upsell Dashboard API",
    description="API to process customer data and serve dashboard insights.",
    version="1.0.0"
)

# Allow the React frontend (running on port 3000) to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Dashboard API!"}

# Include the API router from endpoints.py
app.include_router(api_router, prefix="/api")