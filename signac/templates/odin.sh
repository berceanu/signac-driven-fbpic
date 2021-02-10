{% extends "slurm.sh" %}

{% block tasks %}
{% set cores_per_node = 48 %}
{% set gpus_per_node = 16 %}
{% set threshold = 0 if force else 0.9 %}
{% set cpu_tasks = operations|calc_tasks('np', parallel, force) %}
{% set gpu_tasks = operations|calc_tasks('ngpu', parallel, force) %}
{% if gpu_tasks and 'gpu' not in partition and not force %}
{% raise "Requesting GPUs requires a gpu partition!" %}
{% endif %}
{% set nn_cpu = cpu_tasks|calc_num_nodes(cores_per_node) if 'gpu' not in partition else cpu_tasks|calc_num_nodes(cores_per_node) %}
{% set nn_gpu = gpu_tasks|calc_num_nodes(gpus_per_node) if 'gpu' in partition else 0 %}
{% set nn = nn|default((nn_cpu, nn_gpu)|max, true) %}
{% if 'gpu' in partition %}
#SBATCH --nodes={{ nn|default(1, true) }}
#SBATCH --ntasks-per-node={{ (gpu_tasks, cpu_tasks)|max }}
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:{{ gpu_tasks }}
{% else %}
#SBATCH --nodes={{ nn }}
#SBATCH --ntasks-per-node={{ (cores_per_node, cpu_tasks)|min }}
{% endif %}
{% endblock tasks %}

{% block header %}
{{ super () -}}
#SBATCH --account=berceanu_a+
{% endblock header %}

{% block project_header %}
{{ super() -}}
module use $HOME/MyModules
module load miniforge3pic/latest

export FBPIC_DISABLE_THREADING=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
{% endblock project_header %}

{% block body %}
{% set cmd_suffix = cmd_suffix|default('') ~ (' &' if parallel else '') %}
{% for operation in operations %}

# {{ "%s"|format(operation) }}
export CUDA_VISIBLE_DEVICES={{ loop.index0 }}
{{ operation.cmd }}{{ cmd_suffix }}
{% if operation.eligible_operations|length > 0 %}
# Eligible to run:
{% for run_op in operation.eligible_operations %}
# {{ run_op.cmd }}
{% endfor %}
{% endif %}
{% if operation.operations_with_unmet_preconditions|length > 0 %}
# Operations with unmet preconditions:
{% for run_op in operation.operations_with_unmet_preconditions %}
# {{ run_op.cmd }}
{% endfor %}
{% endif %}
{% if operation.operations_with_met_postconditions|length > 0 %}
# Operations with all postconditions met:
{% for run_op in operation.operations_with_met_postconditions %}
# {{ run_op.cmd }}
{% endfor %}
{% endif %}
{% endfor %}
{% endblock body %}
