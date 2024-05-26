#!/bin/bash

# Activate the virtual environment
source env/bin/activate

# Run the Python module with provided arguments
python3.8 -m src.cli.run_task "$@"