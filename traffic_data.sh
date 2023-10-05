#!/bin/bash

# Set your token
ACCESS_TOKEN="$TOKEN"


RESPONSE=$(curl -X 'GET' \
  'https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point=52.41072%2C4.84239&unit=KMPH&key='$ACCESS_TOKEN'' \
  -H 'accept: */*')

# Save the result to output.json
echo -e "$RESPONSE" > output.json
