from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    DB_USER: str
    DB_PASS: str
    DB_HOST: str
    DB_PORT: str
    DB_NAME: str

    class Config:
        env_file = '.env'


settings = Settings()

List_coins = ['BTCUSDT',
              'ETHUSDT',
              'ADAUSDT',
              'POLUSDT',
              'LINKUSDT',
              'LTCUSDT',
              'XRPUSDT',
              'AVAXUSDT',
              'ATOMUSDT',
              'ALGOUSDT',
              'XTZUSDT',
              ]