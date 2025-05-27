from fastapi import APIRouter

from app.api.v1.endpoints import auth, blog, projects, contact, subscribers, tags, categories, users

api_router = APIRouter()

# Include all API endpoints
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(blog.router, prefix="/blog", tags=["blog"])
api_router.include_router(projects.router, prefix="/projects", tags=["projects"])
api_router.include_router(contact.router, prefix="/contact", tags=["contact"])
api_router.include_router(subscribers.router, prefix="/subscribers", tags=["subscribers"])
api_router.include_router(tags.router, prefix="/tags", tags=["tags"])
api_router.include_router(categories.router, prefix="/categories", tags=["categories"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
