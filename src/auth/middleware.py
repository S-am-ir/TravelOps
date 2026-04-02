"""
FastAPI auth dependencies — extract and verify JWT from Authorization header.
"""

from typing import Optional
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.auth.service import decode_access_token, AuthService, UserProfile

security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> UserProfile:
    """Require authentication — raises 401 if not logged in."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")

    claims = decode_access_token(credentials.credentials)
    if not claims:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = await AuthService.get_user(claims["sub"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[UserProfile]:
    """Optional auth — returns user if logged in, None otherwise."""
    if not credentials:
        return None

    claims = decode_access_token(credentials.credentials)
    if not claims:
        return None

    return await AuthService.get_user(claims["sub"])
