import argparse
import multiprocessing
import torch


def check_cpu():
    num_cpu_cores = multiprocessing.cpu_count()
    print(f"Number of CPU cores available: {num_cpu_cores}")


def check_gpu():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPU devices available: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPU devices available")


def main():
    parser = argparse.ArgumentParser(description="Check number of cores on GPU or CPU")
    parser.add_argument(
        "--device",
        type=str,
        choices=["gpu", "cpu"],
        required=True,
        help="Device to check cores for (gpu or cpu)",
    )

    args = parser.parse_args()

    if args.device == "gpu":
        check_gpu()
    else:
        check_cpu()


# This is just an example of how to run the python script from the shell script with params
if __name__ == "__main__":
    main()
