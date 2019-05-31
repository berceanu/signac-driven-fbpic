{% set cmd_suffix = cmd_suffix|default('') ~ (' &' if parallel else '') %}
{% for operation in operations %}
export CUDA_VISIBLE_DEVICES={{ loop.index0 }}
{{ operation.cmd }}{{ cmd_suffix }}
{% endfor %}
wait
