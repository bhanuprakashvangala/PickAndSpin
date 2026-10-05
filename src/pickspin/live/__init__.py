"""Live runs of Pick and Spin on a Kubernetes deployment of vLLM servers.

vllm holds the endpoint map and the OpenAI-compatible chat call, actuator scales the model Deployments
(the kubernetes client comes with the [live] extra), and runner wires Pick, Spin, the actuator and a
thread pool into the live experiment. Import from the modules.
"""
