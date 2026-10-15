# Single-host TPU env for one worker of the shared 2x2x4 slice.
# Only code and venvs live in /dev/shm (it counts toward the pod's memory limit); caches,
# logs, tmp and profiles live on shared storage, per slice (the hostname is the session name).
export HOME=/dev/shm/home/qinghanc
export FORCE_SINGLE_HOST=1 TPU_TOPOLOGY=2x2x1 TPU_HOST_BOUNDS=1,1,1 TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 \
       TPU_WORKER_HOSTNAMES=localhost TPU_WORKER_ID=0 TPU_PROCESS_ADDRESSES=localhost:8471 TPU_ACCELERATOR_TYPE=tpu7x-4
SHARED=/work/qinghan; POD_DIR=$SHARED/pods/$(hostname)
export TPU_LOG_DIR=$POD_DIR/tpu_logs XDG_CACHE_HOME=$POD_DIR/cache TMPDIR=$POD_DIR/tmp \
       HF_HOME=$SHARED/cache/hf TORCH_TPU_TIER3_ROOT=$SHARED/torch_tpu_compile_cache \
       SGLANG_CACHE_DIR=$POD_DIR/cache/sglang
mkdir -p $TPU_LOG_DIR $XDG_CACHE_HOME $TMPDIR $HF_HOME $POD_DIR/glm-run $POD_DIR/prof
[ -f $HOME/venv/bin/activate ] && . $HOME/venv/bin/activate
