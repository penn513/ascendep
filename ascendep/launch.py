"""Local torchrun launcher with bounded lifetime and explicit loopback store."""
import os
import platform
import signal
import socket
import subprocess
import sys


def run_local(arguments, module=False, timeout=150, capture_output=False, env=None, cwd=None):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    environment = dict(os.environ if env is None else env)
    environment.setdefault("OMP_NUM_THREADS", "1")
    if platform.system() == "Darwin":
        # The libuv TCPStore rendezvous failed on the validated macOS host;
        # the legacy store and explicit loopback interface passed the probe.
        environment.setdefault("USE_LIBUV", "0")
        environment.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    command = [sys.executable, "-m", "torch.distributed.run", "--master-addr=127.0.0.1",
               f"--master-port={port}", "--nproc-per-node=2"]
    if module:
        command.append("--module")
    command += list(arguments)
    process = subprocess.Popen(command, env=environment, cwd=cwd, text=True, start_new_session=True,
                               stdout=subprocess.PIPE if capture_output else None,
                               stderr=subprocess.PIPE if capture_output else None)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        raise
    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    result.check_returncode()
    return result
