#/bin/bash

# TF Example:
## Node 1
# TF_CONIG='{"cluster": {"worker": ["10.233.68.1:5005", "10.233.66.110:5005"]}, "task": {"index": 0, "type": "worker"}}' python worker.py
## Node 2
# TF_CONIG='{"cluster": {"worker": ["10.233.68.1:5005", "10.233.66.110:5005"]}, "task": {"index": 1, "type": "worker"}}' python worker.py
clear
export APP_NAME=tensorflow-distributed
export WORKER_IPS=$(kubectl get pods -l run=$APP_NAME-workers -o jsonpath='{range .items[*]}{.status.podIP}{":5005 "}{end}' | xargs)
export PORT=5005
export INDEX=0

GRPC_ARRAY=()
for IP in $WORKER_IPS;
    do
        GRPC_ARRAY+="$IP,"
    done

export GRPC_ARRAY=$(sed 's/,*$//g' <<< $GRPC_ARRAY)

for i in $WORKER_IPS;
do
    TF_CONFIG=$( jq -c -n \
                    --arg workers "$GRPC_ARRAY" \
                    --arg index "$INDEX" \
                    '{cluster: {worker: $workers|split(",")}, task: {index: $index, type: "worker"}}'
                    )
    echo "TF_CONFIG='$TF_CONFIG' python3 worker.py"

    INDEX=$(($INDEX+1));
done