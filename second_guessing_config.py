"""Configuration for the SecondGuessingGPT project."""
# pylint: disable=wrong-import-position
import os

from dotenv import load_dotenv

load_dotenv()

import promptlayer

DJANGO_SECRET_KEY = os.environ["DJANGO_SECRET_KEY"]

DJANGO_DEBUG = (os.environ.get("DJANGO_DEBUG") or "false").lower() in ("true", "1", "yes", "y")
DJANGO_LOG_LEVEL = os.environ.get("DJANGO_LOG_LEVEL") or "INFO"
SECOND_GUESSING_LOG_LEVEL = os.environ.get("SECOND_GUESSING_LOG_LEVEL") or "INFO"

DJANGO_HOST = os.environ["DJANGO_HOST"]
DJANGO_BASE_URL = f"https://{DJANGO_HOST}"
TELEGRAM_WEBHOOK_PATH = "telegram_webhook"

TELEGRAM_WEBHOOK_TOKEN = os.environ["TELEGRAM_WEBHOOK_TOKEN"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

pl_async_openai_client = promptlayer.openai.AsyncOpenAI()
