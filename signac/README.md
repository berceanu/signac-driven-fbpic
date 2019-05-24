# Usage

```console
conda activate signac-driven-fbpic

python3 src/init.py
python3 src/project.py run # serial version 
# parallel on 6 GPUs
python3 src/project.py submit --bundle=6 --parallel --test | /bin/bash
# python3 src/project.py run --parallel
python3 src/project.py status --pretty --full --stack
python3 src/dashboard.py run --host 0.0.0.0 --port 7777
```
