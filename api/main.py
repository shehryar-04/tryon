import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from api.config import settings
from api.routers import tryon, health

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise-grade Virtual Try-On API for E-Commerce Platforms (Shopify, WooCommerce, Custom)",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for e-commerce cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount local static storage if local backend is selected
if settings.STORAGE_BACKEND == "local":
    os.makedirs(settings.LOCAL_STORAGE_DIR, exist_ok=True)
    app.mount("/static", StaticFiles(directory=settings.LOCAL_STORAGE_DIR), name="static")

# Mount test harness demo widget
widget_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "widget")
if os.path.exists(widget_dir):
    app.mount("/widget", StaticFiles(directory=widget_dir, html=True), name="widget")

# Register API Routers
app.include_router(health.router, prefix=settings.API_V1_PREFIX)
app.include_router(tryon.router, prefix=settings.API_V1_PREFIX)

@app.get("/")
async def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "demo_widget": "/widget/index.html",
        "status": "operational"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)

