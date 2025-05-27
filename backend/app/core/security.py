from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Union

from jose import jwt
from passlib.context import CryptContext

from app.core.config import settings


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ALGORITHM = "HS256"


def create_access_token(
    subject: Union[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode = {"exp": expire, "sub": str(subject)}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against a hash
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password
    """
    return pwd_context.hash(password)


def decode_access_token(token: str) -> Optional[dict]:
    """
    Decode a JWT access token
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        # Token has expired
        return None
    except jwt.JWTError:
        # Any other JWT error (e.g., invalid signature, malformed token)
        return None

# Example of how to use in an endpoint dependency:
# from fastapi import Depends, HTTPException, status
# from fastapi.security import OAuth2PasswordBearer
# from app.core.security import decode_access_token
# from app.schemas.schemas import TokenData # Assuming you have this schema
# from app.models.models import User # Assuming you have this model
# from app.api.deps import get_db # Assuming you have this dependency
# from sqlalchemy.orm import Session

# oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/token")

# async def get_current_user(
#     db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)
# ) -> User:
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     payload = decode_access_token(token)
#     if payload is None:
#         raise credentials_exception
#     username: str = payload.get("sub")
#     if username is None:
#         raise credentials_exception
#     token_data = TokenData(username=username)
    
#     user = db.query(User).filter(User.username == token_data.username).first() # Adjust query as per your User model
#     if user is None:
#         raise credentials_exception
#     return user

# async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
#     if current_user.disabled: # Assuming your User model has a 'disabled' attribute
#         raise HTTPException(status_code=400, detail="Inactive user")
#     return current_user

# async def get_current_active_superuser(current_user: User = Depends(get_current_user)) -> User:
#     if not current_user.is_superuser: # Assuming your User model has an 'is_superuser' attribute
#         raise HTTPException(
#             status_code=403, detail="The user doesn't have enough privileges"
#         )
#     return current_user
