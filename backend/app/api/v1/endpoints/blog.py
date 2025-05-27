from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Any, List, Optional
import shutil
from pathlib import Path

from app import schemas, models
from app.api import deps
from app.core.config import settings
from app.utils import generate_slug # Assuming you will create this utility

router = APIRouter()

# Helper function to save uploaded file
async def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()

@router.post("/", response_model=schemas.BlogPost, status_code=status.HTTP_201_CREATED)
def create_blog_post(
    *, # Enforces keyword-only arguments
    db: Session = Depends(deps.get_db),
    title: str = Form(...),
    summary: Optional[str] = Form(None),
    content: str = Form(...),
    category_name: Optional[str] = Form(None), # Create/get category by name
    tag_names: Optional[List[str]] = Form(None), # Create/get tags by name
    image: Optional[UploadFile] = File(None),
    featured: bool = Form(False),
    published_at: Optional[datetime] = Form(None),
    read_time_minutes: Optional[int] = Form(None),
    current_user: models.User = Depends(deps.get_current_active_superuser) # Or get_current_active_user
) -> Any:
    """
    Create new blog post.
    Requires superuser privileges.
    Category and Tags can be created on the fly if they don't exist.
    """
    slug = generate_slug(title) # Implement this utility function
    if db.query(models.BlogPost).filter(models.BlogPost.slug == slug).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A blog post with this title (slug) already exists."
        )

    db_category = None
    if category_name:
        db_category = db.query(models.Category).filter(models.Category.name == category_name).first()
        if not db_category:
            category_slug = generate_slug(category_name)
            db_category = models.Category(name=category_name, slug=category_slug)
            db.add(db_category)
            # db.commit() # Commit separately or together later
            # db.refresh(db_category)

    db_tags = []
    if tag_names:
        for tag_name in tag_names:
            db_tag = db.query(models.Tag).filter(models.Tag.name == tag_name).first()
            if not db_tag:
                tag_slug = generate_slug(tag_name)
                db_tag = models.Tag(name=tag_name, slug=tag_slug)
                db.add(db_tag)
                # db.commit() # Commit separately or together later
                # db.refresh(db_tag)
            db_tags.append(db_tag)

    image_url_path = None
    if image:
        # Ensure uploads directory exists (config should handle this, but double check)
        settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        # Create a unique filename to avoid overwrites
        file_extension = Path(image.filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        image_save_path = settings.UPLOADS_DIR / unique_filename
        await save_upload_file(image, image_save_path)
        image_url_path = f"/uploads/{unique_filename}" # Relative path for serving

    blog_post_data = schemas.BlogPostCreate(
        title=title,
        slug=slug, # Will be auto-generated if not provided or based on title
        summary=summary,
        content=content,
        author_id=current_user.id,
        category_id=db_category.id if db_category else None,
        # tags will be handled separately after post creation or via association
        image_url=image_url_path,
        featured=featured,
        published_at=published_at,
        read_time_minutes=read_time_minutes,
        tag_names=[] # Placeholder, tags are linked below
    )
    
    # Exclude fields not in BlogPost model directly for creation
    blog_post_model_data = blog_post_data.model_dump(exclude={"tag_names"})
    
    db_blog_post = models.BlogPost(**blog_post_model_data)
    db_blog_post.tags = db_tags # Assign the tag objects
    
    db.add(db_blog_post)
    db.commit()
    db.refresh(db_blog_post)
    return db_blog_post

@router.get("/", response_model=List[schemas.BlogPost])
def read_blog_posts(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    category: Optional[str] = None,
    tag: Optional[str] = None,
    featured: Optional[bool] = None,
    search: Optional[str] = None,
    # current_user: models.User = Depends(deps.get_current_active_user) # Optional: if posts are user-specific or need auth
) -> Any:
    """
    Retrieve blog posts. 
    Optionally filter by category slug, tag slug, featured status, or search query.
    """
    query = db.query(models.BlogPost).order_by(models.BlogPost.published_at.desc(), models.BlogPost.created_at.desc())

    if category:
        query = query.join(models.Category).filter(models.Category.slug == category)
    
    if tag:
        query = query.join(models.BlogPost.tags).filter(models.Tag.slug == tag)

    if featured is not None:
        query = query.filter(models.BlogPost.featured == featured)

    if search:
        # Basic search in title and summary. For advanced search, consider Elasticsearch/OpenSearch.
        search_term = f"%{search}%"
        query = query.filter(
            models.BlogPost.title.ilike(search_term) |
            models.BlogPost.summary.ilike(search_term) # Add content if needed, but can be slow
        )
    
    # Only show published posts to non-superusers or if not admin view
    # if not (current_user and current_user.is_superuser):
    #     query = query.filter(models.BlogPost.published_at <= datetime.now(timezone.utc))
    #     query = query.filter(models.BlogPost.published_at != None)

    total_posts = query.count() # Get total count before skip/limit for pagination headers if needed
    blog_posts = query.offset(skip).limit(limit).all()
    
    # Could add X-Total-Count header for pagination here
    return blog_posts

@router.get("/{post_id_or_slug}", response_model=schemas.BlogPost)
def read_blog_post(
    post_id_or_slug: str,
    db: Session = Depends(deps.get_db),
    # current_user: models.User = Depends(deps.get_current_active_user) # Optional auth
) -> Any:
    """
    Get blog post by ID or slug.
    """
    try:
        # Try to convert to UUID first for ID lookup
        post_uuid = uuid.UUID(post_id_or_slug)
        db_blog_post = db.query(models.BlogPost).filter(models.BlogPost.id == post_uuid).first()
    except ValueError:
        # If not a valid UUID, assume it's a slug
        db_blog_post = db.query(models.BlogPost).filter(models.BlogPost.slug == post_id_or_slug).first()

    if db_blog_post is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Blog post not found")
    
    # Optional: Check if post is published if user is not admin
    # if not (current_user and current_user.is_superuser) and \
    #    (not db_blog_post.published_at or db_blog_post.published_at > datetime.now(timezone.utc)):
    #     raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Blog post not found or not published")
        
    return db_blog_post

@router.put("/{post_id}", response_model=schemas.BlogPost)
def update_blog_post(
    post_id: uuid.UUID,
    *, # Enforces keyword-only arguments
    db: Session = Depends(deps.get_db),
    title: Optional[str] = Form(None),
    summary: Optional[str] = Form(None),
    content: Optional[str] = Form(None),
    category_name: Optional[str] = Form(None),
    tag_names: Optional[List[str]] = Form(None),
    image: Optional[UploadFile] = File(None),
    delete_image: Optional[bool] = Form(False), # Flag to delete existing image
    featured: Optional[bool] = Form(None),
    published_at: Optional[datetime] = Form(None),
    read_time_minutes: Optional[int] = Form(None),
    current_user: models.User = Depends(deps.get_current_active_superuser)
) -> Any:
    """
    Update a blog post. Requires superuser privileges.
    """
    db_blog_post = db.query(models.BlogPost).filter(models.BlogPost.id == post_id).first()
    if not db_blog_post:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Blog post not found")

    update_data = {}
    if title is not None:
        update_data["title"] = title
        new_slug = generate_slug(title)
        if new_slug != db_blog_post.slug:
            if db.query(models.BlogPost).filter(models.BlogPost.slug == new_slug).first():
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="New title creates a slug conflict.")
            update_data["slug"] = new_slug
            
    if summary is not None: update_data["summary"] = summary
    if content is not None: update_data["content"] = content
    if featured is not None: update_data["featured"] = featured
    if published_at is not None: update_data["published_at"] = published_at
    if read_time_minutes is not None: update_data["read_time_minutes"] = read_time_minutes

    if category_name is not None:
        if category_name == "": # Explicitly remove category
            update_data["category_id"] = None
        else:
            db_category = db.query(models.Category).filter(models.Category.name == category_name).first()
            if not db_category:
                category_slug = generate_slug(category_name)
                db_category = models.Category(name=category_name, slug=category_slug)
                db.add(db_category)
                # db.commit() # Consider committing changes together
                # db.refresh(db_category)
            update_data["category_id"] = db_category.id

    if tag_names is not None:
        new_tags = []
        if tag_names: # If list is not empty
            for tag_name in tag_names:
                db_tag = db.query(models.Tag).filter(models.Tag.name == tag_name).first()
                if not db_tag:
                    tag_slug = generate_slug(tag_name)
                    db_tag = models.Tag(name=tag_name, slug=tag_slug)
                    db.add(db_tag)
                    # db.commit()
                    # db.refresh(db_tag)
                new_tags.append(db_tag)
        db_blog_post.tags = new_tags # Replace existing tags

    if delete_image and db_blog_post.image_url:
        # Delete old image file from server
        old_image_path_str = db_blog_post.image_url.replace("/uploads/", "", 1) if db_blog_post.image_url else None
        if old_image_path_str:
            old_image_path = settings.UPLOADS_DIR / old_image_path_str
            if old_image_path.exists():
                old_image_path.unlink()
        update_data["image_url"] = None
        
    if image:
        # Delete old image if a new one is uploaded
        if db_blog_post.image_url and not delete_image: # only delete if not already marked for deletion
            old_image_path_str = db_blog_post.image_url.replace("/uploads/", "", 1)
            old_image_path = settings.UPLOADS_DIR / old_image_path_str
            if old_image_path.exists():
                old_image_path.unlink()
        
        settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        file_extension = Path(image.filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        image_save_path = settings.UPLOADS_DIR / unique_filename
        await save_upload_file(image, image_save_path)
        update_data["image_url"] = f"/uploads/{unique_filename}"

    for field, value in update_data.items():
        setattr(db_blog_post, field, value)
    
    db.add(db_blog_post)
    db.commit()
    db.refresh(db_blog_post)
    return db_blog_post

@router.delete("/{post_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_blog_post(
    post_id: uuid.UUID,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_active_superuser)
) -> None:
    """
    Delete a blog post. Requires superuser privileges.
    Also deletes the associated image file if it exists.
    """
    db_blog_post = db.query(models.BlogPost).filter(models.BlogPost.id == post_id).first()
    if not db_blog_post:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Blog post not found")

    # Delete associated image file
    if db_blog_post.image_url:
        image_path_str = db_blog_post.image_url.replace("/uploads/", "", 1)
        image_path = settings.UPLOADS_DIR / image_path_str
        if image_path.exists():
            image_path.unlink()
            
    db.delete(db_blog_post)
    db.commit()
    return None # No content to return

# You might want to add endpoints for categories and tags as well
# e.g., GET /categories, GET /tags, POST /categories, etc.
# For simplicity, they are managed implicitly via blog post creation/update here.

# Example: Get all categories
@router.get("/utils/categories", response_model=List[schemas.Category])
def read_categories(db: Session = Depends(deps.get_db)) -> Any:
    categories = db.query(models.Category).order_by(models.Category.name).all()
    return categories

# Example: Get all tags
@router.get("/utils/tags", response_model=List[schemas.Tag])
def read_tags(db: Session = Depends(deps.get_db)) -> Any:
    tags = db.query(models.Tag).order_by(models.Tag.name).all()
    return tags


# Placeholder for slug generation utility - should be in app/utils.py
import re
from slugify import slugify as pyslugify # Using python-slugify library

def generate_slug(text: str) -> str:
    # Basic slugification, can be enhanced (e.g., check for uniqueness in DB)
    return pyslugify(text)

# Add this to app/api/deps.py or a new app/utils.py
# from app.models.models import User as UserModel # Renamed to avoid conflict
# from app.core.security import decode_access_token
# from app.schemas.schemas import TokenPayload
# from fastapi.security import OAuth2PasswordBearer

# oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/token")

# def get_current_user(
#     db: Session = Depends(get_db),
#     token: str = Depends(oauth2_scheme)
# ) -> UserModel:
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     payload = decode_access_token(token)
#     if not payload or not payload.get("sub"):
#         raise credentials_exception
#     token_data = TokenPayload(sub=payload["sub"])
    
#     user_id_str = str(token_data.sub)
#     try:
#         user_id = uuid.UUID(user_id_str)
#     except ValueError:
#         raise credentials_exception # Invalid UUID format for user ID

#     user = db.query(UserModel).filter(UserModel.id == user_id).first()
#     if user is None:
#         raise credentials_exception
#     return user

# def get_current_active_user(current_user: UserModel = Depends(get_current_user)) -> UserModel:
#     if not current_user.is_active:
#         raise HTTPException(status_code=400, detail="Inactive user")
#     return current_user

# def get_current_active_superuser(current_user: UserModel = Depends(get_current_active_user)) -> UserModel:
#     if not current_user.is_superuser:
#         raise HTTPException(
#             status_code=403, detail="The user doesn't have enough privileges"
#         )
#     return current_user

# Ensure deps.py has these functions or move them there.
# For now, defining a placeholder for generate_slug here if not in utils.

