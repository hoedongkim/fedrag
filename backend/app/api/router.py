"""The /api router that mounts every route module."""

from fastapi import APIRouter

from app.api.routes import ask, compare, evaluation, health, meetings, stance

api_router = APIRouter(prefix="/api")
for module in (health, meetings, ask, stance, compare, evaluation):
    api_router.include_router(module.router)
