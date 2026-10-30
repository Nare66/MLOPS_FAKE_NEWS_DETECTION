import os
from fastapi import Header, HTTPException, status, Depends
from backend.app.config import settings

def verify_admin_key(
    x_admin_key: str = Header(None, alias="X-Admin-Key"),
    authorization: str = Header(None, alias="Authorization")
):
    """
    Validates administrative access for protected MLOps and continuous learning endpoints.
    Accepts either X-Admin-Key header or Bearer token.
    """
    token = x_admin_key
    if not token and authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]
        elif len(parts) == 1:
            token = parts[0]

    if not token or token != settings.ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Valid Administrative API Key required for this operation.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return True
