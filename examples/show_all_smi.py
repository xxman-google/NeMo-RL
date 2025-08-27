"""Show GPU usage from all nodes.

nvidia-smi only shows one node.
"""

import ray
import subprocess
import socket

@ray.remote
def get_smi_output():
    try:
        return (socket.gethostname(), subprocess.check_output(['nvidia-smi'], text=True))
    except Exception as e:
        return (socket.gethostname(), f"Failed to run nvidia-smi: {e}")

def show_all_smi():
    ray.init(address='auto', ignore_reinit_error=True)
    tasks = [
        get_smi_output.options(resources={f"node:{n['NodeManagerAddress']}": 0.01}).remote()
        for n in ray.nodes() if n['alive']
    ]
    for host, output in sorted(ray.get(tasks)):
        print(f"--- NVIDIA-SMI on {host} ---\n{output}")

if __name__ == '__main__':
    show_all_smi()