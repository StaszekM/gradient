import papermill as pm
import os

os.chdir("/home/staszek/mgr/gradient/gradient/")

pm.execute_notebook(
    "notebooks/baseline_multiple_cities/baseline_multiple_cities.ipynb",
    "notebooks/baseline_multiple_cities/out/krakow.ipynb",
    parameters=dict(
        city_name="Kraków", nominatim_city_name="Kraków, Poland", year=2022
    ),
)


pm.execute_notebook(
    "notebooks/baseline_multiple_cities/baseline_multiple_cities.ipynb",
    "notebooks/baseline_multiple_cities/out/warszawa.ipynb",
    parameters=dict(
        city_name="Warszawa", nominatim_city_name="Warszawa, Poland", year=2022
    ),
)


pm.execute_notebook(
    "notebooks/baseline_multiple_cities/baseline_multiple_cities.ipynb",
    "notebooks/baseline_multiple_cities/out/poznan.ipynb",
    parameters=dict(
        city_name="Poznań", nominatim_city_name="Poznań, Poland", year=2022
    ),
)


pm.execute_notebook(
    "notebooks/baseline_multiple_cities/baseline_multiple_cities.ipynb",
    "notebooks/baseline_multiple_cities/out/szczecin.ipynb",
    parameters=dict(
        city_name="Szczecin", nominatim_city_name="Szczecin, Poland", year=2022
    ),
)
