#/bin/bash
NGC_EMAIL=btuttle@nvidia.com
NGC_API_KEY=MmFiMWplNnNqbzZscWZxOWFnczY5M3ZocmE6YjY1MzkyMjUtMzZmMC00NzUwLWI2ZTItODhlOTAyMGFjNjQ3
WEATHER_STACK_API_KEY=77ff5f0502b4290d086dfbb2eacf6226

helm delete distributed-training --wait
sudo rm -rf /export/deepops_nfs/default-tensorflow-distributed-pvc-pvc-2ae5abce-ed25-454e-a295-c4bd3ecf311c/tensorboard
echo 
ls -l /export/deepops_nfs/default-tensorflow-distributed-pvc-pvc-2ae5abce-ed25-454e-a295-c4bd3ecf311c/
echo 
helm install distributed-training distributed-training \
    --set imageCredentials.password=$NGC_API_KEY \
    --set imageCredentials.email=$NGC_EMAIL