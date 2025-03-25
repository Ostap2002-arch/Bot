from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy import create_engine

from config import settings

DB_URL = f"postgresql+asyncpg://{settings.DB_USER}:{settings.DB_PASS}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
engine = create_async_engine(DB_URL,
                             pool_size=30,
                             max_overflow=20,
                             pool_timeout=60,  # Ждать соединение до 30 сек
                             pool_recycle=300,  # Пересоздавать соединения каждые 5 мин (избегаем "умерших" соединений)
                             pool_pre_ping=True,  # Проверять соединение перед использованием
                             # Таймауты
                             echo=False)

session_maker = sessionmaker(bind=engine,
                             class_=AsyncSession,
                             expire_on_commit=False,
                             )
