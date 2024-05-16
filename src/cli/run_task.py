import yaml
from pprint import pprint
from argparse import ArgumentParser
import os
import papermill as pm
import sys

parser = ArgumentParser()

parser.add_argument(
    "--notebook-output",
    type=str,
    required=True,
    help="Decide where to output the notebook.",
)
parser.add_argument(
    "--wandb-key",
    type=str,
    required=False,
    help="Wandb key for logging. Used only in some downstream tasks.",
)
args = parser.parse_args()

if args.wandb_key:
    os.environ["WANDB_API_KEY"] = args.wandb_key

with open("config.yml", "r") as stdout_file:
    parameters = yaml.safe_load(stdout_file)

print("\nRun parameters:")
pprint(parameters)

downstream_task = parameters["downstream_task"]
with open(f"downstream_task_config/{downstream_task}.yml", "r") as stdout_file:
    downstream_task_parameters = yaml.safe_load(stdout_file)
print("\nDownstream task parameters:")
pprint(downstream_task_parameters, depth=2)

print("\nInput notebook location:")
notebook_location = (
    os.getcwd() + "/notebooks/downstream_tasks/" + downstream_task + ".ipynb"
)
print(notebook_location)


pm.execute_notebook(
    input_path=notebook_location,
    output_path=args.notebook_output,
    parameters=downstream_task_parameters["params"],
    progress_bar=False,
    log_output=True,
    stdout_file=sys.stdout,
)
