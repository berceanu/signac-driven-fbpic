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
#SBATCH --ntasks={{ (gpu_tasks, cpu_tasks)|max }}
#SBATCH --ntasks-per-node={{ (gpu_tasks, cpu_tasks)|max }}
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=31200m
#SBATCH --gres=gpu:{{ gpu_tasks }}
#SBATCH --gres-flags=enforce-binding
{% else %}
#SBATCH --nodes={{ nn }}
#SBATCH --ntasks-per-node={{ (cores_per_node, cpu_tasks)|min }}
{% endif %}
{% endblock tasks %}

{% block header %}
{{ super () -}}
#SBATCH --account=berceanu_a+
#SBATCH --export=HOME,USER,TERM,WRKDIR
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

{{ operation.cmd }}{{ cmd_suffix }}
{% endfor %}
{% endblock body %}
