import requests
import os
import json
import random


# Define the boundaries of Świdnicka Street in Wroclaw using latitude and longitude
swidnicka_street_bounds = {
    "min_lat": 51.103724,
    "max_lat": 51.114358,
    "min_lon": 17.024212,
    "max_lon": 17.040517
}

# Generate random latitude and longitude
random_lat = random.uniform(swidnicka_street_bounds["min_lat"], swidnicka_street_bounds["max_lat"])
random_lon = random.uniform(swidnicka_street_bounds["min_lon"], swidnicka_street_bounds["max_lon"])

# Print random coordinates
print("Random Coordinates:")
print("Latitude:", random_lat)
print("Longitude:", random_lon)


api_key = os.environ["API_KEY"]


# Send a request to the tomtom website for the traffic information
url = (
    f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?key={api_key}"
    f"&point={random_lat},{random_lon}")
response = requests.get(url)

if response.status_code == 200:
    traffic_data_json = response.json()
    filename = "traffic_data.json"
    with open(filename, "w") as json_file:
        json.dump(traffic_data_json, json_file, indent=4)

    print(f"Traffic data saved to {filename}")

else:
    print("Failed to retrieve data. Status code:", response.status_code)