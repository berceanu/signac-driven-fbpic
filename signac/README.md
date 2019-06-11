# Usage

```console
conda activate signac-driven-fbpic

#python3 src/init.py
./init.sh

# parallel on N GPUs
./project.sh N

# check completion status while running
python3 src/project.py status --pretty --full --stack

# launch web interface for result visualization
python3 src/dashboard.py run --host 0.0.0.0 --port 7777
```

# Accessing the web interface from your local machine

- launch the web interface on the server, inside a `screen` session

- create a `ssh` tunnel on your local machine, via
  ```console
  $ ssh -f cetal -L 9999:localhost:7777 -N
  ```
  where `cetal` is a host present in your ` ~/.ssh/config`

- go to `localhost:9999` in your web browser
