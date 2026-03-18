from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Twilio
    TWILIO_ACCOUNT_SID: str
    TWILIO_AUTH_TOKEN: str
    TWILIO_WHATSAPP_NUMBER: str

    # Google Maps
    GOOGLE_MAPS_API_KEY: str

    # Database
    DB_HOST: str = "216.48.190.41"
    DB_PORT: int = 5432
    DB_NAME: str = "cab_logistics_test"
    DB_USER: str = "cab_db_user_rw"
    DB_PASSWORD: str

    # Local LLM (vLLM)
    VLLM_BASE_URL: str = "http://localhost:8000/v1"
    VLLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"

    # Local Whisper
    WHISPER_MODEL: str = "large-v3-turbo"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
