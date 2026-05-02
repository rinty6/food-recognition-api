"""
FatSecret Platform API client (OAuth 2.0 client-credentials flow).

Endpoints used:
  foods.search.v3  — find best FatSecret food match for a class name
  food.get.v4      — fetch serving/nutrition details for a food_id

Results are cached in memory for 24 hours to minimise API calls.

Set FATSECRET_CLIENT_ID and FATSECRET_CLIENT_SECRET in .env or Railway env vars.
"""

import os
import time
import hashlib
import logging
from typing import Any

import httpx
from dotenv import load_dotenv, find_dotenv

from api.schemas import NutritionInfo

# find_dotenv() walks up parent directories, so it finds .env in machine_learning/
# or any ancestor folder even when uvicorn is started from food_recognition/
load_dotenv(find_dotenv(usecwd=False, raise_error_if_not_found=False))

logger = logging.getLogger(__name__)

TOKEN_URL  = "https://oauth.fatsecret.com/connect/token"
API_URL    = "https://platform.fatsecret.com/rest/server.api"
CACHE_TTL  = 86_400  # 24 hours in seconds

# Food-101 class names use underscores; FatSecret searches better with spaces
def _class_to_query(class_name: str) -> str:
    return class_name.replace("_", " ")


class FatSecretClient:
    def __init__(self):
        self.client_id     = os.getenv("FATSECRET_CLIENT_ID", "")
        self.client_secret = os.getenv("FATSECRET_CLIENT_SECRET", "")
        self._token: str | None = None
        self._token_expires: float = 0.0
        self._cache: dict[str, tuple[float, NutritionInfo | None]] = {}

    def _is_configured(self) -> bool:
        configured = bool(self.client_id and self.client_secret)
        if not configured:
            logger.warning(
                "FATSECRET_CLIENT_ID or FATSECRET_CLIENT_SECRET not set. "
                "Nutrition data will be null. Check your .env file."
            )
        return configured

    async def _get_token(self) -> str:
        if self._token and time.time() < self._token_expires:
            return self._token

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                TOKEN_URL,
                data={
                    "grant_type":    "client_credentials",
                    "client_id":     self.client_id,
                    "client_secret": self.client_secret,
                    "scope":         "premier",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        self._token = data["access_token"]
        self._token_expires = time.time() + data.get("expires_in", 86400) - 60
        return self._token

    async def _search_food_id(self, query: str) -> str | None:
        token = await self._get_token()
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                API_URL,
                headers={"Authorization": f"Bearer {token}"},
                data={
                    "method":            "foods.search.v5",
                    "search_expression": query,
                    "format":            "json",
                    "max_results":       "1",
                    "page_number":       "0",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        logger.info(f"FatSecret search top-level keys: {list(data.keys())}")

        # FatSecret error response (HTTP 200 but body contains error)
        if "error" in data:
            logger.warning(f"FatSecret search error: {data['error']}")
            return None

        # v3 format: foods_search.results.food
        foods = data.get("foods_search", {}).get("results", {}).get("food")
        # v0/v1 format: foods.food
        if foods is None:
            foods = data.get("foods", {}).get("food")

        logger.info(f"FatSecret foods value: {foods}")

        if not foods:
            return None
        first = foods[0] if isinstance(foods, list) else foods
        food_id = first.get("food_id")
        logger.info(f"FatSecret food_id found: {food_id}")
        return str(food_id) if food_id else None

    async def _get_nutrition(self, food_id: str) -> NutritionInfo | None:
        token = await self._get_token()
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                API_URL,
                headers={"Authorization": f"Bearer {token}"},
                data={
                    "method":   "food.get.v5",
                    "food_id":  food_id,
                    "format":   "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        logger.info(f"FatSecret food.get top-level keys: {list(data.keys())}")
        if "error" in data:
            logger.warning(f"FatSecret food.get error: {data['error']}")
            return None

        food = data.get("food", {})
        servings = food.get("servings", {}).get("serving", [])
        logger.info(f"FatSecret servings value type={type(servings).__name__}, value={servings}")
        if not servings:
            return None

        # Use the first serving (typically 1 serving or 100g)
        s = servings[0] if isinstance(servings, list) else servings

        def _f(key: str) -> float | None:
            try:
                return float(s[key])
            except (KeyError, TypeError, ValueError):
                return None

        return NutritionInfo(
            calories=_f("calories"),
            protein_g=_f("protein"),
            carbs_g=_f("carbohydrate"),
            fat_g=_f("fat"),
            fiber_g=_f("fiber"),
            serving_description=s.get("serving_description"),
        )

    async def get_nutrition_for_class(self, class_name: str) -> NutritionInfo | None:
        """Main entry point. Returns cached NutritionInfo or fetches from FatSecret."""
        if not self._is_configured():
            logger.debug("FatSecret credentials not set — skipping nutrition lookup")
            return None

        cache_key = class_name
        cached = self._cache.get(cache_key)
        if cached:
            expires_at, info = cached
            if time.time() < expires_at:
                return info

        try:
            query   = _class_to_query(class_name)
            food_id = await self._search_food_id(query)
            if not food_id:
                info = None
            else:
                info = await self._get_nutrition(food_id)
        except Exception as e:
            logger.warning(f"FatSecret lookup failed for '{class_name}': {e}")
            info = None

        self._cache[cache_key] = (time.time() + CACHE_TTL, info)
        return info


# Module-level singleton
_client: FatSecretClient | None = None


def get_fatsecret_client() -> FatSecretClient:
    global _client
    if _client is None:
        _client = FatSecretClient()
    return _client
