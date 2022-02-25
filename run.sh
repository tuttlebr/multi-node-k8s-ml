#/bin/bash

export APP_NAME=tensorflow-distributed
export WORKER_IPS=$(kubectl get pods -l run=$APP_NAME-workers -o jsonpath='{range .items[*]}{.status.podIP}{":4242 "}{end}' | xargs)
export PORT=4242
export INDEX=0
export GRPC_ARRAY=$(sed 's/ /\, /g' <<< $WORKER_IPS)

for i in $WORKER_IPS;

do
    kubectl exec pod/$APP_NAME-workers-$INDEX -- sh -c "TF_CONFIG='{"cluster": {"worker": [$GRPC_ARRAY]}, "task": {"index": $INDEX, "type": "worker"}}'; python3 worker.py" &
    INDEX=$(($INDEX+1));
done