from datetime import datetime
from typing import List, Optional, Union
import uuid

from pydantic import BaseModel, EmailStr, Field, HttpUrl, validator
from slugify import slugify


# Token Schemas
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenPayload(BaseModel):
    sub: Optional[Union[uuid.UUID, str]] = None  # Changed from id to sub to match JWT standard


# Base User Schemas
class UserBase(BaseModel):
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    is_active: Optional[bool] = True
    is_superuser: bool = False


class UserCreate(UserBase):
    password: str


class UserUpdate(UserBase):
    password: Optional[str] = None


class UserInDBBase(UserBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True  # Replaces orm_mode = True


class User(UserInDBBase):  # For returning user data
    pass


class UserInDB(UserInDBBase):  # For internal use, includes hashed_password
    hashed_password: str


# Base Category Schemas
class CategoryBase(BaseModel):
    name: str
    slug: str
    description: Optional[str] = None


class CategoryCreate(CategoryBase):
    pass


class CategoryUpdate(CategoryBase):
    name: Optional[str] = None
    slug: Optional[str] = None


class Category(CategoryBase):
    id: uuid.UUID
    created_at: datetime

    class Config:
        from_attributes = True


# Base Tag Schemas
class TagBase(BaseModel):
    name: str
    slug: str


class TagCreate(TagBase):
    pass


class TagUpdate(TagBase):
    name: Optional[str] = None
    slug: Optional[str] = None


class Tag(TagBase):
    id: uuid.UUID
    created_at: datetime

    class Config:
        from_attributes = True


# Base Blog Post Schemas
class BlogPostBase(BaseModel):
    title: str
    slug: str
    summary: Optional[str] = None
    content: str
    category_id: Optional[uuid.UUID] = None
    image_url: Optional[str] = None  # Changed HttpUrl to str for flexibility
    featured: bool = False
    published_at: Optional[datetime] = None
    read_time_minutes: Optional[int] = None


class BlogPostCreate(BlogPostBase):
    author_id: uuid.UUID  # Must be provided on creation
    tag_names: Optional[List[str]] = []  # Allow creating/linking tags by name


class BlogPostUpdate(BlogPostBase):
    title: Optional[str] = None
    slug: Optional[str] = None
    content: Optional[str] = None
    tag_names: Optional[List[str]] = None  # Allow updating tags by name


class BlogPost(BlogPostBase):
    id: uuid.UUID
    author: User  # Nested User schema
    category: Optional[Category] = None  # Nested Category schema
    tags: List[Tag] = []  # Nested Tag schemas
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class BlogPostList(BaseModel):
    total: int
    items: List[BlogPost]


# Base Project Schemas
class ProjectBase(BaseModel):
    title: str
    slug: str
    description: str
    category_id: Optional[uuid.UUID] = None
    technologies: Optional[List[str]] = []
    highlights: Optional[List[str]] = []
    status: Optional[str] = "In Development"
    image_url: Optional[str] = None  # Changed HttpUrl to str for flexibility
    github_url: Optional[str] = None  # Changed HttpUrl to str for flexibility
    live_url: Optional[str] = None  # Changed HttpUrl to str for flexibility
    featured: bool = False
    display_order: Optional[int] = 0


class ProjectCreate(ProjectBase):
    pass


class ProjectUpdate(ProjectBase):
    title: Optional[str] = None
    slug: Optional[str] = None
    description: Optional[str] = None


class Project(ProjectBase):
    id: uuid.UUID
    category: Optional[Category] = None  # Nested Category schema
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ProjectList(BaseModel):
    total: int
    items: List[Project]


# Contact Submission Schemas
class ContactSubmissionBase(BaseModel):
    name: str
    email: EmailStr
    subject: Optional[str] = None
    message: str


class ContactSubmissionCreate(ContactSubmissionBase):
    pass


class ContactSubmission(ContactSubmissionBase):
    id: uuid.UUID
    submitted_at: datetime
    is_read: bool = False

    class Config:
        from_attributes = True


# Newsletter Subscription Schemas
class NewsletterSubscriptionBase(BaseModel):
    email: EmailStr


class NewsletterSubscriptionCreate(NewsletterSubscriptionBase):
    pass


class NewsletterSubscription(NewsletterSubscriptionBase):
    id: uuid.UUID
    subscribed_at: datetime
    is_active: bool = True

    class Config:
        from_attributes = True


# Generic message schema for responses
class Message(BaseModel):
    message: str
