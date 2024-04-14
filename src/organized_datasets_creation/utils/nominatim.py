"""Nominatim-related utilities module.

"""

import unidecode
import requests
from typing import Dict

nominatim_cache: Dict[str, str] = {}


def convert_nominatim_name_to_filename(name: str) -> str:
    """Prepares the name of the city to be used as a filename.

    The process involves:
    - Unidecoding non-ASCII characters
    - Replacing spaces with underscores
    - Converting the name to lowercase
    """
    return unidecode.unidecode(name).replace(" ", "_").lower()


def resolve_nominatim_city_name(name: str) -> str:
    """Asks Nominatim API for the city name and returns it.

    Each call during the session caches the result in a dictionary.
    If the requested name is already in the cache, it returns the value from the cache, without hitting the API.

    Parameters
    ----------
    name : str
        The name that will be passed as a query parameter in the URL:

        `https://nominatim.openstreetmap.org/search?q={name}&format=json`

    Returns
    -------
    str
        Nominatim name of the city.
    """
    cached_name = nominatim_cache.get(name)

    if cached_name:
        return cached_name

    value = requests.get(
        f"https://nominatim.openstreetmap.org/search?q={name}&format=json"
    ).json()[0]

    nominatim_cache[name] = value.get("name")

    return value.get("name")
