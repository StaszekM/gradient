from .nominatim import (
    convert_nominatim_name_to_filename,
    resolve_nominatim_city_name,
)


def create_dataset_location(
    nominatim_city_name: str, resolution: int, method: str, year: int
) -> str:
    """Creates the dataset location as a hierarchical structure of directories.

    Parameters
    ----------
    nominatim_city_name : str
        Name of the city that should comply to the Nominatim format and be present in the accidents.csv file.
        For proper name resolution, use the `resolve_nominatim_city_name` function.
        More info on Nominatim [here](https://nominatim.openstreetmap.org/ui/search.html).
    resolution : int
        Any resolution that is used in the dataset creation process.
    method : str
        The identifier of the embedding method used in the dataset creation process, e.g. "count-embedder", "hex2vec", "highway2vec".
    year : int
        Year which the dataset is created for (e.g. what year the accidents are from)

    Returns
    -------
    str
        The resulting string with the dataset location.

    Examples
    --------

    >>> create_dataset_location("Warszawa, Poland", 6, "count-embedder", 2017)
    'warszawa/2017/h6/count-embedder'
    """
    city_name_as_filename = convert_nominatim_name_to_filename(
        resolve_nominatim_city_name(nominatim_city_name)
    )

    return f"{city_name_as_filename}/{year}/h{resolution}/{method}"
