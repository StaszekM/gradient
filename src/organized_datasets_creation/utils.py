import unidecode
import requests
from typing import Dict

nominatim_cache: Dict[str, str] = {}


def convert_nominatim_name_to_filename(name: str) -> str:
    return unidecode.unidecode(name).replace(" ", "_").lower()


def resolve_nominatim_city_name(name: str) -> str:
    cached_name = nominatim_cache.get(name)

    if cached_name:
        return cached_name

    value = requests.get(
        f"https://nominatim.openstreetmap.org/search?q={name}&format=json"
    ).json()[0]

    nominatim_cache[name] = value.get("name")

    return value.get("name")
