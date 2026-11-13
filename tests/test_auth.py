"""Sign in with Google, the allowlist, the session cookie and the daily limit.

Google's real check is replaced by a fake that accepts tokens like
"valid:friend@gmail.com", so no real Google account is needed.
"""

import pytest
from fastapi.testclient import TestClient

import api
import auth

QUESTION = {"question": "How do I load filament?"}


def fake_verify(credential):
    if not credential.startswith("valid:"):
        raise ValueError("Token used too late")  # like a real expired token
    email = credential.removeprefix("valid:")
    return {"email": email, "email_verified": True, "name": "Test User", "picture": ""}


@pytest.fixture
def login_on(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client.apps.googleusercontent.com")
    monkeypatch.setenv("SESSION_SECRET", "x" * 40)
    monkeypatch.setenv("ALLOWED_EMAILS", "owner@gmail.com, Friend@Gmail.com")
    monkeypatch.setattr(auth, "verify_google_token", fake_verify)


@pytest.fixture
def web(services, login_on):
    with TestClient(api.create_app(services)) as test_client:
        yield test_client


def sign_in(client, email):
    return client.post("/api/auth/google", json={"credential": f"valid:{email}"})


# --- Signing in ---

def test_everything_needs_a_login_except_health_and_config(web):
    assert web.get("/api/health").status_code == 200
    assert web.get("/api/auth/config").json() == {
        "google_client_id": "test-client.apps.googleusercontent.com"
    }
    assert web.get("/api/documents").status_code == 401
    assert web.post("/api/ask", json=QUESTION).status_code == 401


def test_sign_in_sets_a_session_cookie(web):
    response = sign_in(web, "owner@gmail.com")
    assert response.status_code == 200
    assert response.json()["email"] == "owner@gmail.com"

    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie       # JavaScript can't read it
    assert "samesite=lax" in cookie   # other sites can't send it with a POST

    me = web.get("/api/me").json()
    assert me["email"] == "owner@gmail.com"
    assert me["login_enabled"] is True
    assert web.get("/api/documents").status_code == 200


def test_allowlist_ignores_upper_and_lower_case(web):
    assert sign_in(web, "FRIEND@gmail.com").json()["email"] == "friend@gmail.com"


def test_emails_not_on_the_list_are_refused(web):
    response = sign_in(web, "stranger@gmail.com")
    assert response.status_code == 403
    assert "access list" in response.json()["detail"]
    assert web.get("/api/documents").status_code == 401  # no session was created


def test_fake_or_expired_google_tokens_are_refused(web):
    response = web.post("/api/auth/google", json={"credential": "made-up"})
    assert response.status_code == 401
    assert web.get("/api/documents").status_code == 401


def test_removing_someone_from_the_list_locks_them_out_at_once(web, monkeypatch):
    sign_in(web, "friend@gmail.com")
    assert web.get("/api/documents").status_code == 200

    monkeypatch.setenv("ALLOWED_EMAILS", "owner@gmail.com")
    assert web.get("/api/documents").status_code == 403
    assert web.get("/api/documents").status_code == 401  # and the session is gone


def test_sign_out(web):
    sign_in(web, "owner@gmail.com")
    assert web.post("/api/auth/logout").status_code == 204
    assert web.get("/api/documents").status_code == 401


def test_a_forged_cookie_is_ignored(web):
    web.cookies.set("support_rag_session", "eyJlbWFpbCI6ICJvd25lckBnbWFpbC5jb20ifQ==.fake.signature")
    assert web.get("/api/documents").status_code == 401


def test_secure_cookies_on_the_live_site(services, login_on, monkeypatch):
    monkeypatch.setenv("COOKIE_SECURE", "true")
    with TestClient(api.create_app(services)) as client:
        assert "secure" in sign_in(client, "owner@gmail.com").headers["set-cookie"].lower()


def test_login_needs_a_long_session_secret(services, login_on, monkeypatch):
    monkeypatch.setenv("SESSION_SECRET", "short")
    with pytest.raises(RuntimeError, match="SESSION_SECRET"):
        api.create_app(services)


def test_live_site_refuses_to_start_without_login_settings(services, monkeypatch):
    # fail closed: better down than open to everyone
    monkeypatch.setenv("REQUIRE_LOGIN", "true")
    with pytest.raises(RuntimeError, match="GOOGLE_CLIENT_ID"):
        api.create_app(services)


def test_without_google_settings_everyone_is_the_local_user(client):
    me = client.get("/api/me").json()
    assert me["email"] == auth.LOCAL_USER_EMAIL
    assert me["login_enabled"] is False
    assert client.post("/api/auth/google", json={"credential": "valid:x@y.com"}).status_code == 400


# --- Daily question limit ---

def test_daily_limit_per_user(web, monkeypatch):
    monkeypatch.setenv("DAILY_QUESTION_LIMIT", "2")
    sign_in(web, "friend@gmail.com")
    assert web.post("/api/ask", json=QUESTION).status_code == 200
    assert web.get("/api/me").json()["questions_left_today"] == 1
    assert web.post("/api/ask", json=QUESTION).status_code == 200

    response = web.post("/api/ask", json=QUESTION)
    assert response.status_code == 429
    assert "2 questions for today" in response.json()["detail"]

    # a different person still has their own allowance
    sign_in(web, "owner@gmail.com")
    assert web.post("/api/ask", json=QUESTION).status_code == 200


def test_model_errors_do_not_use_up_the_limit(web, fake_gemini, make_pdf, monkeypatch):
    monkeypatch.setenv("DAILY_QUESTION_LIMIT", "1")
    sign_in(web, "friend@gmail.com")
    web.post("/api/documents", files={"file": ("m.pdf", make_pdf(["Loading filament: push it in."]), "application/pdf")})
    fake_gemini.generator.error = type("Busy", (Exception,), {"status_code": 503})()

    assert web.post("/api/ask", json={"question": "Loading filament push"}).status_code == 503
    assert web.get("/api/me").json()["questions_left_today"] == 1
