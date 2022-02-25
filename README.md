# Multi-node ML With Kubernetes
End-to-end deployment for multi-node training using GPU nodes on a Kubernetes cluster.


## Intro
This helm chart will deploy a StatefulSet of N replicas as defined in the chart's values.yaml. This is where resources and affinity may be defined and allows for a generic port to most programming languages

## Prerequisite
1. Kubernetes Cluster (Tested on >= v1.20)
2. Two or more nodes with at least one NVIDIA GPU (Tested on Amphere architecture)
3. NFS Access for shared storage
4. [NVIDIA NGC Account](https://catalog.ngc.nvidia.com/)


## Install

```bash
helm install tensorflow distributed-training \
    --set imageCredentials.password=<NGC_API_KEY> \
    --set imageCredentials.email=<NGC_USER_EMAIL> \
    --set replicaCount=<N_NODES> \
    --set resources.limits.nvidia.com/gpu=<GPUS_PER_NODE> \
    --set tensorboardNode=<SINGE_NODE_HOSTNAME> \
    --set nfs.path=<NFS_PATH_WITH_YOUR_DATA> \
    --set nfs.server=<NFS_SERVER_IP_WITH_YOUR_DATA>
```

Tensorflow Config
Set environment variable `TF_CONFIG` on each node and begin training.
Helper function is included and may be modified. Other modifications may be needed for packages other than TensorFlow.
```bash
./run.sh
```