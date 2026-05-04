"""
USDA FoodData Central API client.

Replaces FatSecret — no IP restrictions, free API key, covers raw foods.
Register at: https://fdc.nal.usda.gov/api-key-signup.html

Set USDA_API_KEY in Railway environment variables (or .env for local dev).
Results are cached in memory for 24 hours.
"""

import os
import time
import logging
from typing import Any

import httpx
from dotenv import load_dotenv, find_dotenv

from api.schemas import NutritionInfo

load_dotenv(find_dotenv(usecwd=False, raise_error_if_not_found=False))

logger = logging.getLogger(__name__)

BASE_URL = "https://api.nal.usda.gov/fdc/v1"
CACHE_TTL = 86_400  # 24 hours

# USDA nutrient IDs we care about
_NID_CALORIES = 1008
_NID_PROTEIN  = 1003
_NID_CARBS    = 1005
_NID_FAT      = 1004
_NID_FIBER    = 1079

# Food-101 class names use underscores; USDA searches better with spaces
def _class_to_query(class_name: str) -> str:
    return class_name.replace("_", " ")


class UsdaClient:
    def __init__(self):
        self.api_key = os.getenv("USDA_API_KEY", "")
        self._cache: dict[str, tuple[float, NutritionInfo | None]] = {}

    def _is_configured(self) -> bool:
        if not self.api_key:
            logger.warning(
                "USDA_API_KEY not set. Nutrition data will be null. "
                "Get a free key at https://fdc.nal.usda.gov/api-key-signup.html"
            )
            return False
        return True

    async def _search(self, query: str) -> NutritionInfo | None:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{BASE_URL}/foods/search",
                params={
                    "query": query,
                    "api_key": self.api_key,
                    # Foundation and SR Legacy have the most complete raw nutrient data
                    "dataType": "Foundation,SR Legacy",
                    "pageSize": 1,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        foods = data.get("foods", [])
        if not foods:
            # Retry without dataType filter to catch branded / Survey foods
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{BASE_URL}/foods/search",
                    params={"query": query, "api_key": self.api_key, "pageSize": 1},
                )
                resp.raise_for_status()
                data = resp.json()
            foods = data.get("foods", [])

        if not foods:
            logger.info(f"USDA: no results for '{query}'")
            return None

        food = foods[0]
        nutrients: dict[int, float] = {}
        for n in food.get("foodNutrients", []):
            nid = n.get("nutrientId")
            val = n.get("value")
            if nid and val is not None:
                nutrients[nid] = float(val)

        # Serving description: prefer the food's portion info, fall back to "100g"
        serving_desc = food.get("servingSize") and (
            f"{food['servingSize']} {food.get('servingSizeUnit', 'g')}"
        )
        if not serving_desc:
            serving_desc = "100 g"

        logger.info(f"USDA: matched '{food.get('description')}' for query '{query}'")

        return NutritionInfo(
            calories=nutrients.get(_NID_CALORIES),
            protein_g=nutrients.get(_NID_PROTEIN),
            carbs_g=nutrients.get(_NID_CARBS),
            fat_g=nutrients.get(_NID_FAT),
            fiber_g=nutrients.get(_NID_FIBER),
            serving_description=serving_desc,
        )

    async def get_nutrition_for_class(self, class_name: str) -> NutritionInfo | None:
        """Main entry point — same signature as FatSecretClient for easy swap."""
        if not self._is_configured():
            return None

        cached = self._cache.get(class_name)
        if cached:
            expires_at, info = cached
            if time.time() < expires_at:
                return info

        try:
            query = _class_to_query(class_name)
            info = await self._search(query)
        except Exception as e:
            logger.warning(f"USDA lookup failed for '{class_name}': {e}")
            info = None

        self._cache[class_name] = (time.time() + CACHE_TTL, info)
        return info


_client: UsdaClient | None = None


def get_usda_client() -> UsdaClient:
    global _client
    if _client is None:
        _client = UsdaClient()
    return _client
