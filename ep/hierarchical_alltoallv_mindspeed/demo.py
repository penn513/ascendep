"""Forward the multi-node feature flags to an existing MindSpeed r0.8 training driver."""
import argparse
import os
import subprocess
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("driver")
    parser.add_argument("driver_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", 1)) <= int(os.environ.get("LOCAL_WORLD_SIZE", 1)):
        parser.exit(2, "Unsupported on one host: hierarchical AllToAllV requires multiple nodes.\n")
    subprocess.run([sys.executable, args.driver, *args.driver_args, "--moe-hierarchical-alltoallv",
                    "--moe-tp-extend-ep", "--moe-alltoall-overlap-comm"], check=True)
