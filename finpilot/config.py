import os
from dotenv import load_dotenv

load_dotenv()

MESH_BASE_URL = "https://api.meshapi.ai/v1"
MODEL = os.getenv("FINPILOT_MODEL", "openai/gpt-4o-mini")
DB_URL = os.getenv("FINPILOT_DB", "sqlite:///finpilot.db")

CATEGORIES = ["rent", "groceries", "dining", "food_delivery", "transport", "utilities",
              "subscriptions", "shopping", "entertainment", "health", "travel",
              "income", "savings", "other"]
