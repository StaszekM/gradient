"""Nominatim-related utilities module.

"""

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


import unidecode
import requests
from typing import Dict

nominatim_cache: Dict[str, str] = {
    "Wrocław, Poland": "Wrocław",
    "Warszawa, Poland": "Warszawa",
    "Szczecin, Poland": "Szczecin",
    "Poznań, Poland": "Poznań",
    "Kraków, Poland": "Kraków",
}


def send_get_request(URL):
    sess = requests.session()

    retries = Retry(
        total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
    )

    sess.mount("https://", HTTPAdapter(max_retries=retries))
    get_URL = sess.get(URL, timeout=10)
    return get_URL


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

    value = send_get_request(
        f"https://nominatim.openstreetmap.org/search?q={name}&format=json"
    ).json()

    value = value[0]

    nominatim_cache[name] = value.get("name")

    return value.get("name")
