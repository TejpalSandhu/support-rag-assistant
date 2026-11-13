"""
Sign in with Google, an email allowlist, and a session cookie.

How a login works:
1. The website shows Google's sign-in button. Google gives the browser an
   "ID token": a short message saying "this is friend@gmail.com", signed by
   Google.
2. The browser sends it to POST /api/auth/google. We check Google's
   signature (so nobody can fake one) and that the email is on ALLOWED_EMAILS.
3. We answer with a session cookie signed with SESSION_SECRET. The browser
   sends it with every request for 7 days; JavaScript can't read it
   (HttpOnly), and other websites can't make the browser send it (SameSite).

Settings (environment variables):
    GOOGLE_CLIENT_ID   from Google Cloud Console; if empty, login is off and
                       everyone is "local@localhost" (local development)
    SESSION_SECRET     a long random string that signs the cookies
    ALLOWED_EMAILS     comma-separated emails that may use the app
    COOKIE_SECURE      "true" on the live site: cookie only sent over HTTPS
    REQUIRE_LOGIN      "true" on the live site: refuse to start without the
                       settings above, so a config mistake can't leave the
                       site open to everyone
"""

import os
import secrets
from dataclasses import dataclass

from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

SESSION_DAYS = 7
LOCAL_USER_EMAIL = "local@localhost"


@dataclass
class User:
    email: str
    name: str = ""
    picture: str = ""


def google_client_id() -> str:
    return os.getenv("GOOGLE_CLIENT_ID", "").strip()


def login_enabled() -> bool:
    return bool(google_client_id())


def allowed_emails() -> set[str]:
    return {
        email.strip().lower()
        for email in os.getenv("ALLOWED_EMAILS", "").split(",")
        if email.strip()
    }


def verify_google_token(credential: str) -> dict:
    """Check Google's signature, the audience (our app) and expiry; return its claims.

    Raises ValueError for anything fake, expired or meant for another app.
    """
    claims = id_token.verify_oauth2_token(
        credential, google_requests.Request(), audience=google_client_id(),
    )
    if not claims.get("email_verified"):
        raise ValueError("Google says this email address isn't verified.")
    return claims


def current_user(request: Request) -> User:
    """Dependency for every protected endpoint: who is asking?"""
    if not login_enabled():
        return User(email=LOCAL_USER_EMAIL, name="Local user")

    email = request.session.get("email")
    if not email:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Please sign in.")
    # checked on every request, so removing someone takes effect at once
    if email not in allowed_emails():
        request.session.clear()
        raise HTTPException(status.HTTP_403_FORBIDDEN, f"{email} no longer has access.")
    return User(email, request.session.get("name", ""), request.session.get("picture", ""))


def add_sessions(app: FastAPI) -> None:
    if os.getenv("REQUIRE_LOGIN", "").lower() == "true" and not login_enabled():
        raise RuntimeError("REQUIRE_LOGIN is set but GOOGLE_CLIENT_ID is missing.")
    secret = os.getenv("SESSION_SECRET", "")
    if login_enabled() and len(secret) < 32:
        raise RuntimeError("Set SESSION_SECRET to a random string of at least 32 characters.")
    app.add_middleware(
        SessionMiddleware,
        # without login there's nothing worth protecting: a throwaway secret is fine
        secret_key=secret or secrets.token_hex(32),
        session_cookie="support_rag_session",
        max_age=SESSION_DAYS * 24 * 3600,
        same_site="lax",
        https_only=os.getenv("COOKIE_SECURE", "").lower() == "true",
    )


# --- Endpoints (no login needed to reach these) ---

router = APIRouter(prefix="/api/auth")


class GoogleLogin(BaseModel):
    credential: str


@router.get("/config")
def config() -> dict[str, str | None]:
    # the client ID isn't secret: it's in every Google sign-in button
    return {"google_client_id": google_client_id() or None}


@router.post("/google")
def sign_in(body: GoogleLogin, request: Request) -> User:
    if not login_enabled():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Google sign-in isn't set up on this server.")
    try:
        claims = verify_google_token(body.credential)
    except ValueError as error:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Google sign-in failed: {error}")

    email = claims["email"].lower()
    if email not in allowed_emails():
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"{email} isn't on the access list. Ask the owner to add you.",
        )
    user = User(email, claims.get("name", ""), claims.get("picture", ""))
    request.session.clear()  # a fresh session for every login
    request.session.update(email=user.email, name=user.name, picture=user.picture)
    return user


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def sign_out(request: Request) -> None:
    request.session.clear()
