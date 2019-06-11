# Usage

```console
conda activate signac-driven-fbpic

#python3 src/init.py
./init.sh

# serial version for testing purposes
python3 src/project.py run 

# parallel on N GPUs
./project.sh N

# check completion status while running
python3 src/project.py status --pretty --full --stack

# launch web interface for result visualization
python3 src/dashboard.py run --host 0.0.0.0 --port 7777
```
