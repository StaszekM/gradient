import papermill as pm
import os

os.chdir('/Users/jjaniak/Documents/studia/projekt/gradient')


pm.execute_notebook(
    "notebooks/accident_analysis/eda_other_cities.ipynb",
    "notebooks/accident_analysis/out/warszawa.ipynb",
    parameters=dict(
        city_name="Warszawa", nominatim_city_name="Warszawa, Poland", year=2022
    ),
)


pm.execute_notebook(
    "notebooks/accident_analysis/eda_other_cities.ipynb",
    "notebooks/accident_analysis/out/poznan.ipynb",
    parameters=dict(
        city_name="Poznań", nominatim_city_name="Poznań, Poland", year=2022
    ),
)


pm.execute_notebook(
    "notebooks/accident_analysis/eda_other_cities.ipynb",
    "notebooks/accident_analysis/out/szczecin.ipynb",
    parameters=dict(
        city_name="Szczecin", nominatim_city_name="Szczecin, Poland", year=2022
    ),
)
