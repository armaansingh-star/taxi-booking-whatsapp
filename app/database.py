import asyncpg
import logging

from app.config import settings

log = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


async def init_pool(min_size: int = 2, max_size: int = 10):
    global _pool
    _pool = await asyncpg.create_pool(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        database=settings.DB_NAME,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        min_size=min_size,
        max_size=max_size,
    )
    log.info("Database pool initialized (min=%d, max=%d)", min_size, max_size)


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        log.info("Database pool closed")


def get_pool() -> asyncpg.Pool:
    assert _pool is not None, "Database pool not initialized. Call init_pool() first."
    return _pool
