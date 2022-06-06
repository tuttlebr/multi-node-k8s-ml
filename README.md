# Multi-node ML With Kubernetes

End-to-end deployment for multi-node training using GPU nodes on a Kubernetes cluster.

## Intro

This helm chart will deploy a StatefulSet of N replicas as defined in the chart's values.yaml. This is where resources and affinity may be defined and allows for a generic port to most programming languages

## Prerequisite

1. Kubernetes Cluster (Tested on >= v1.20)
2. Two or more nodes with at least one NVIDIA GPU per node (Tested on Amphere architecture)
3. [NVIDIA NGC Account](https://catalog.ngc.nvidia.com/)
4. [NGC command line tool]
5. Pytorch Example: `ngc registry resource download-version "nvidia/bert_for_pytorch:20.06.18" --dest pytorch`
6. Tensorflow Example: `ngc registry resource download-version "nvidia/bert_for_tensorflow:20.06.17" --dest tensorflow`

## Install

Using the `distributed-training/values.yaml` file, set the parameters as needed for your cluster. The most notable options will be the node affinity and resources. Node affinity will allocate all pods to nodes of your defined affinity. Resources will allocate all resources per pod as your defined resources.

```bash
helm install tensorflow distributed-training \
    --set imageCredentials.password=<NGC_API_KEY> \
    --set imageCredentials.email=<NGC_USER_EMAIL>
```
