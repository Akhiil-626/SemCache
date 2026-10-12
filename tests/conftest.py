from collections.abc import AsyncGenerator
import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.main import app


@pytest.fixture(autouse=True)
def configure_test_environment():
    """Ensures environment is set to 'test' during pytest runs."""
    original_env = settings.ENVIRONMENT
    settings.ENVIRONMENT = "test"
    yield
    settings.ENVIRONMENT = original_env


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Provides an asynchronous HTTP test client for the FastAPI application."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
