import argparse
from pathlib import Path
import sys

from azureml.core import Workspace, Environment, Experiment, ScriptRunConfig, Dataset
from azureml.data import OutputFileDatasetConfig

sys.path.append(str(Path('./tfm')))
from tfm.constants import SAMPLER, VERSION

parser = argparse.ArgumentParser()
parser.add_argument('--task', choices=[1, 2, 4, 5], type=int, required=True)
args, extra_args = parser.parse_known_args()

ws = Workspace.from_config()
target = ws.compute_targets['cpu-cluster-4-28']
dataset = Dataset.File.from_files(
    path=(ws.datastores['tfm'], 'google_data')
)
output = OutputFileDatasetConfig(
    destination=(ws.datastores['tfm'], "out"),
    # name="out",
)
env = Environment.from_pip_requirements(
    'tfm-env',
    'requirements.txt'
)
version_str = "{}.{}.{}".format(*sys.version_info)
env.python.conda_dependencies.set_python_version(version_str)
exp = Experiment(ws, f'tfm-exp-task{args.task}-{SAMPLER.name}{"-" if VERSION else ""}{VERSION}')
config = ScriptRunConfig(
    source_directory='tfm',
    command=[
        'python', 'main.py', dataset.as_mount(), '-o', output.as_mount(),
            '--task', str(args.task), '--timeout', '500', *extra_args,
        '&&', 'curl' ,'-d', 'Finished experiment', 'ntfy.sh/tfm_tda_exp',
        '||', 'curl' ,'-d', 'Failed experiment', 'ntfy.sh/tfm_tda_exp',
    ],
    compute_target=target,
    environment=env
)

run = exp.submit(config)
print(run)
run.wait_for_completion(show_output=True)
