from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config.settings import get_settings

settings = get_settings()

# Create engine with connection pooling
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,          # Persistent connections in pool
    max_overflow=20,       # Extra connections under load
    pool_timeout=30,       # Wait time for connection from pool
    pool_recycle=3600,     # Recycle connections after 1 hour (prevents stale)
)

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Dependency for database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Async database configuration — ready for async endpoints
# Usage: from config.database import async_session
#   async with async_session() as session: ...
try:
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    _async_url = settings.DATABASE_URL.replace(
        "postgresql://", "postgresql+asyncpg://"
    ).replace(
        "postgresql+psycopg2://", "postgresql+asyncpg://"
    )
    async_engine = create_async_engine(
        _async_url,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        pool_recycle=3600,
    )
    async_session = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
except ImportError:
    # asyncpg not installed — async endpoints won't work but sync is fine
    async_engine = None
    async_session = None

