import os
import papermill as pm
from utils import (
    convert_nominatim_name_to_filename,
    resolve_nominatim_city_name,
)


resolutions = [6, 7, 8, 9, 10, 11]
methods = ["count-embedder", "hex2vec", "highway2vec"]
cities = [
    "Warszawa, Poland",
    "Wrocław, Poland",
    "Poznań, Poland",
    "Kraków, Poland",
    "Szczecin, Poland",
]
years = [2017, 2018, 2019, 2020, 2021, 2022]

notebook_location = "/home/staszek/mgr/gradient/gradient/data/organized-datasets/"
root_project_location = "/home/staszek/mgr/gradient/gradient/"


for resolution in resolutions:
    for method in methods:
        for year in years:
            for city in cities:
                city_as_filename = convert_nominatim_name_to_filename(
                    resolve_nominatim_city_name(city)
                )

                if not os.path.exists(
                    os.path.join(
                        notebook_location,
                        f"notebooks_out/{city_as_filename}/h{resolution}/{method}/{year}",
                    )
                ):
                    os.makedirs(
                        os.path.join(
                            notebook_location,
                            f"notebooks_out/{city_as_filename}/h{resolution}/{method}/{year}",
                        )
                    )

                pm.execute_notebook(
                    os.path.join(notebook_location, "download.ipynb"),
                    os.path.join(
                        notebook_location,
                        f"notebooks_out/{city_as_filename}/h{resolution}/{method}/{year}/out.ipynb",
                    ),
                    parameters=dict(
                        resolution=resolution,
                        method=method,
                        year=year,
                        notebook_location=notebook_location,
                        nominatim_city_name=city,
                        root_project_location=root_project_location,
                    ),
                )
