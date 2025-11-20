import http
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.routing import APIRoute
from httpx import Request
from starlette.responses import JSONResponse

from app.core.config import settings
from app.core.exceptions import AppException
from app.utils.model_loader import model_loader

from app.api.routes import predict


def custom_generate_unique_id(route: APIRoute) -> str:
    tag = route.tags[0] if route.tags else "Default"
    return f"{tag} - {route.name}"

@asynccontextmanager
async def lifespan(app:FastAPI):
    logging.info("Loading  up models...")
    model_loader.load_models()
    yield
    logging.warn("Shutting down application")
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_VERSION}/openapi.json",
    generate_unique_id_function=custom_generate_unique_id,
    version="0.0.1",
    contact={"name": "Aham Kingsley", "email": "kingsleyaham6@gmail.com"},
    lifespan=lifespan
)

app.include_router(predict.router, prefix="/predict", tags=["Disease Predictions"])


# Global error handler for unexpected errors
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    logging.error(f"unhandled error: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code if exc.status_code else http.HTTPStatus.INTERNAL_SERVER_ERROR,
        content={"success": False, "error": exc.message if exc.message else MESSAGE.INTERNAL_SERVER_ERROR}
    )


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Plant Disease Detection API is running",
        "status": "active",
        "supported_plants": ["maize", "cassava"]
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
