"""
app/auth.py
-----------
HTTP Basic authentication layer.

Kept in its own module so both app/main.py and app/routers/admin.py can
import `verify_admin` without creating a circular dependency.

Admin credentials are hardcoded in RAM only — never written to disk or DB.
`secrets.compare_digest` is used for timing-safe comparison to prevent
timing-oracle credential enumeration attacks.
"""

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import os
from dotenv import load_dotenv

load_dotenv()
# Single shared security scheme instance (reused across all endpoints)
security = HTTPBasic()

# Credentials stored in process memory only
_ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME")
_ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD")


def verify_admin(credentials: HTTPBasicCredentials = Depends(security)) -> None:
    """
    FastAPI dependency. Raises HTTP 401 for any non-admin credential pair.
    Attach with:  `_: None = Depends(verify_admin)`
    """
    username_ok = secrets.compare_digest(
        credentials.username.encode("utf-8"),
        _ADMIN_USERNAME.encode("utf-8"),
    )
    password_ok = secrets.compare_digest(
        credentials.password.encode("utf-8"),
        _ADMIN_PASSWORD.encode("utf-8"),
    )
    if not (username_ok and password_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials. Admin access required.",
            headers={"WWW-Authenticate": "Basic"},
        )
