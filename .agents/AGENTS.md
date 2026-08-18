# CDK Benchmark Rules
- Always and only use the actual, real-time output logs of the `bench-client` container from the GKE pod as the sole source of truth for benchmark success criteria and performance metrics.
- Never estimate, simulate, or assume performance metrics. If the benchmark is still compiling or running, wait for it to complete and print the actual metrics from the container logs.
