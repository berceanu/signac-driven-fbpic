{# The following variables are available to all scripts. #}
{% if parallel %}
{% set np_global = operations|map(attribute='directives.np')|sum %}
{% else %}
{% set np_global = operations|map(attribute='directives.np')|max %}
{% endif %}
{% block header %}
#!/bin/bash
#SBATCH --job-name="{{ id }}"
{% if partition %}
#SBATCH --partition={{ partition }}
{% endif %}
{% if memory %}
#SBATCH --mem={{ memory }}
{% endif %}
{% if walltime %}
#SBATCH -t {{ walltime|format_timedelta }}
{% endif %}
{% if job_output %}
#SBATCH --output={{ job_output }}
#SBATCH --error={{ job_output }}
{% endif %}

{# https://docs.signac.io/projects/flow/en/latest/supported_environments/comet.html #}
{% block tasks %}
{% set threshold = 0 if force else 0.9 %}
{% set cpu_tasks = operations|calc_tasks('np', parallel, force) %}
{% set gpu_tasks = operations|calc_tasks('ngpu', parallel, force) %}
{% set nn_cpu = cpu_tasks|calc_num_nodes(48) %}
{% set nn_gpu = gpu_tasks|calc_num_nodes(16) %}
{% set nn = nn|default((nn_cpu, nn_gpu)|max, true) %}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={{ (gpu_tasks * 3, cpu_tasks)|max }}
#SBATCH --gres=gpu:{{ (gpu_tasks, 16)|min }}
{% endblock %}

{% endblock %}

{% block project_header %}
set -e
set -u

### . startjob

module use $HOME/MyModules
module load miniforge3pic/latest

cd {{ project.config.project_dir }}
{% endblock %}
{% block body %}
{% set cmd_suffix = cmd_suffix|default('') ~ (' &' if parallel else '') %}
{% for operation in operations %}
export CUDA_VISIBLE_DEVICES={{ loop.index0 }}
sleep 1m
# {{ "%s"|format(operation) }}
{{ operation.cmd }}{{ cmd_suffix }}
{% endfor %}
{% endblock %}
{% block footer %}
{% if parallel %}
wait
{% endif %}

### . endjob
{% endblock %}
