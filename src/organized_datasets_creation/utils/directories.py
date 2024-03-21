from .nominatim import (
    convert_nominatim_name_to_filename,
    resolve_nominatim_city_name,
)


def create_dataset_location(
    nominatim_city_name: str, resolution: int, method: str, year: int
) -> str:
    city_name_as_filename = convert_nominatim_name_to_filename(
        resolve_nominatim_city_name(nominatim_city_name)
    )

    return f"{city_name_as_filename}/{year}/h{resolution}/{method}"
