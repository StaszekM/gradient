import unidecode
import requests
from typing import Dict

nominatim_cache: Dict[str, str] = {}


def convert_nominatim_name_to_filename(name: str) -> str:
    """Joins the name of the city with underscores and converts it to lowercase."""
    return unidecode.unidecode(name).replace(" ", "_").lower()


def resolve_nominatim_city_name(name: str) -> str:
    """Given the name of a city, it returns the name of the city as it is in the Nominatim database.
    If the name is already in the cache, it returns the value from the cache to avoid making a request to the Nominatim API.
    """
    cached_name = nominatim_cache.get(name)

    if cached_name:
        return cached_name

    value = requests.get(
        f"https://nominatim.openstreetmap.org/search?q={name}&format=json"
    ).json()[0]

    nominatim_cache[name] = value.get("name")

    return value.get("name")
