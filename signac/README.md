# Usage

## Initialize the project's workspace

### from scratch

To initialize the `signac` project from scratch:

```console
$ ./init.sh
```

This will create the folder structure corresponding to `src/init.py`, and
**delete** any existing simulation results (see `init.sh` script for details).

### existing data

If, instead of starting from scratch, one wants to add new simulations to an
existing project, `src/init.py` should be used instead of `init.sh`:

```console
$ conda activate signac-driven-fbpic
$ python3 src/init.py
```

## Run the project operations

To run the `fbpic` simulations defined in `src/init.py`, in parallel, using `N`
GPUs and all CPU cores available on the machine:

```console
$ screen -S fbpic
$ [time] ./project.sh N
```

It is convenient to run the project under a `screen` session, as the `fbpic`
simulations might take a few hours to complete. The optional `time` command will
give the total runtime once the project operations are all completed.

## Check completion status

To check the status of currently running operations, one can (periodically)
execute

```console
$ ./status.sh
```

Where available, the command [`nvtop`](https://github.com/Syllo/nvtop) can be
used to check the usage of the machine's GPUs.

## Visualize post-processed results via the web interface

To check the output of running the `signac` project operations, a web server
can be launched via

```console
$ screen -S dashboard
$ ./dashboard.sh
```

One can then open a web browser and point it to `localhost:7777`.

### Accessing the web interface from your local machine

In case the simulations are running on a remote server, it is possible to
visualize the results from one's local machine by using `ssh` local port
forwarding.

On the remote server

- launch the web interface inside a `screen` session, as described above

On the local machine

- create an `ssh` tunnel for port forwarding

  ```console
  $ ssh -f username@remote_server -L 9999:localhost:7777 -N
  ```

- open a web browser and point it to `localhost:9999`

# Notes

- simulation path on `CETAL` machine: `/Date2/andrei/runs/fbpic/signac-driven-fbpic/signac`
