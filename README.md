# Communication Scheduling Experiments in TensorFlow

## Install
```bash
$ pip3 install -r requirements.txt
```

## Requirement
* TensorFlow with Communication Ordering Support: [github.com/xldrx/orderedtf](https://github.com/xldrx/orderedtf)
* A running TF cluster with 1-PS and some workers. (More info [here](https://www.tensorflow.org/deploy/distributed))
* Python3

## How to Run Experiments
0. Start the TF Cluster. Note the master URL (e.g. grpc://1.2.3.4:2222) and number of workers (e.g. 4).
1. Extract the ordering:
```bash
$ python3 0_extract_orders.py masterUri number_of_workers
```
2. Put the `rpc_orders.h` in "tensorflow/core/distributed_runtime/rpc/" and compile the [OrderedTF](https://github.com/xldrx/orderedtf). Restart the TF Cluster.

3. Run the experiences:
```bash
$ python3 1_run_experiments.py masterUri number_of_workers
```
