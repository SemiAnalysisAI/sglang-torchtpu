---
name: cdk-nfs-experiments
description: Run Cloud DevKit (CDK) benchmark jobs on GKE TPU pods using live NFS-mounted code repositories on cdk-vm. Use when creating recipes, launching CDK jobs, or managing NFS-backed codebases.
---

# CDK NFS Experiments Skill

This skill provides step-by-step instructions for launching and managing Cloud DevKit (CDK) benchmark jobs on GKE TPU pods with live NFS-mounted source code directories.

---

## 1. Overview of Architecture

1. **CDK Control Host (`cdk-vm`)**:
   - Central control VM hosting the `cdk` CLI tool and GKE cluster credentials.
   - All `cdk`, `kubectl`, `git`, and filesystem operations targeting NFS repositories **must be run via SSH to `cdk-vm`**:
     ```bash
     ssh cdk-vm "cd /mnt/sglang-nfs/<repo_name> && <COMMAND>"
     ```
2. **NFS Shared Storage (`/mnt/sglang-nfs`)**:
   - Mounted at `/mnt/sglang-nfs` on both `cdk-vm` and GKE pod containers.
   - Allows instant source code changes without rebuilding heavy Docker images.
3. **GKE Pod Containers**:
   - Launched by CDK recipes in the `cloud-devkit-jobs` Kubernetes namespace.
   - Container startup scripts remap Python dependencies to the editable NFS codebase at container initialization.

---

## 2. Using the CDK CLI (No Sudo Required)

Always execute `cdk` commands directly as the standard user (never use `sudo`):

- **List active and recent jobs**:
  ```bash
  ssh cdk-vm "cdk job list"
  ```
- **Create and launch a new job**:
  ```bash
  ssh cdk-vm "cdk job create <recipe_name>"
  ```
- **Delete / stop a job**:
  ```bash
  ssh cdk-vm "cdk job delete <job_id>"
  ```
- **Check job status and pod details**:
  ```bash
  ssh cdk-vm "cdk job status <job_id>"
  ```

---

## 3. Creating and Registering CDK Recipes

CDK resolves job recipes by looking up `<recipe_name>.yaml` in the user's recipe directory on `cdk-vm`:
`/home/chenjincheng/cloud-devkit/recipes/<recipe_name>.yaml`

### Standard Recipe Specification Template

To run experiments with a live NFS code mount for any project (e.g. `sglang-torchtpu`, `vllm-torchtpu`, etc.):

```yaml
apiVersion: jobset.sigs.k8s.io/v1alpha2
kind: JobSet
metadata:
  name: <recipe_name>
spec:
  replicatedJobs:
    - name: server
      template:
        spec:
          template:
            spec:
              volumes:
                # 1. Mount shared NFS volume
                - name: nfs-codebase-volume
                  nfs:
                    server: 10.225.237.178  # Shared NFS Server IP
                    path: /workspace
              containers:
                - name: app-server
                  image: <docker_image_uri>
                  volumeMounts:
                    - name: nfs-codebase-volume
                      mountPath: /mnt/sglang-nfs
                  command: ["/bin/bash", "-c"]
                  args:
                    - |
                      set -eo pipefail
                      export PYTHONUNBUFFERED=1

                      # 2. Remap container package to editable NFS source code
                      echo "Remapping Python packages to editable NFS source code..."
                      cd /mnt/sglang-nfs/<your_repo_name>
                      pip install -e python/ --no-deps

                      # 3. Launch application server
                      bash benchmark/tpu/launch_server.sh
```

---

## 4. Retrieving Logs & Verifying Benchmark Metrics

### Kubernetes Pod Inspection
To inspect running pods associated with a CDK job:
```bash
ssh cdk-vm "kubectl get pods -n cloud-devkit-jobs | grep <job_id>"
```

### Retrieving Container Output Logs
Always use actual container output logs as the sole source of truth for execution status and performance metrics:

```bash
# Pull client container logs directly from GKE pod
ssh cdk-vm "kubectl logs <pod_name> -n cloud-devkit-jobs -c bench-client"

# Pull server container logs directly from GKE pod
ssh cdk-vm "kubectl logs <pod_name> -n cloud-devkit-jobs -c sglang-node"
```
