#!/usr/bin/env python3
"""
server/main.py — FastAPI Standalone Model Server Application Entrypoint
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.api.v1.endpoints import router as v1_router
from server.core.preprocessor import spatial_manager
from server.core.model_loader import model_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=====================================================================")
    print("      STARTING FASTAPI AGRICULTURAL MODEL SERVING MICROSERVICE        ")
    print("=====================================================================")
    print(f" Dataset Ready : {spatial_manager.is_loaded}")
    print(f" Memory Limit  : Max 2048 MB RAM (LRU Model Cap: {model_manager.max_models})")
    print("=====================================================================\n")
    yield
    print("[SERVER] Shutting down Model Server Microservice. Cleaning RAM cache...")
    model_manager.clear_cache()

app = FastAPI(
    title="Myanmar Agricultural Model Serving API",
    description="High-performance, memory-efficient Standalone Model Serving Microservice with LRU Caching, Redis, Fallback Prototypes, and Composite Intelligence.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for cross-origin Backend Requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API v1 Router
app.include_router(v1_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
