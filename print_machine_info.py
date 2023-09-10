import os
import socket

# TODO: Add functions that print json instead
def print_system_info():
    print("Host name:", socket.gethostname())

    gpu_info = [
        item
        for item in os.popen("nvidia-smi | grep Driver")
        .read()
        .split("  ")[:-1]
        if len(item) > 0
    ]

    print("GPU driver:", gpu_info[-2])
    print("CUDA:", gpu_info[-1])

    ncu_version_info = [
        ele
        for ele in os.popen("ncu --version").read().split("\n")
        if len(ele) > 0
    ][-1].strip()
    nsys_version_info = [
        ele
        for ele in os.popen("nsys --version").read().split("\n")
        if len(ele) > 0
    ][-1].strip()
    print("NVIDIA NCU:", ncu_version_info)
    print("NVIDIA NSYS:", nsys_version_info)

    with open("/etc/os-release") as fd:
        for line in fd.readlines():
            if line.startswith("PRETTY_NAME="):
                print("OS:", line.split("=")[-1].strip())

    print("OS kernel version:", os.popen("uname -r").read().strip())
    print(" ")


def print_python_env_info():
    print("Python:", os.popen("python --version").read().strip())
    print("--------------Python packages--------------")
    print(os.popen("pip list").read())
    print("-------------------------------------------")
    print(" ")


def print_conda_envs_info(conda_envs_names: list[str]):
    print("Conda:", os.popen("conda --version").read().strip())
    for name in conda_envs_names:
        print("-------Conda packages (" + name + ")----------")
        print(os.popen("conda list -n " + name).read())
        print("-------------------------------------------")
        print(" ")