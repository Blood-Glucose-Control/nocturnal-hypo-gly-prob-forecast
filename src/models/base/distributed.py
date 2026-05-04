"""
Distributed training utilities for Time Series Foundation Models.

This module provides utilities for setting up and managing distributed training
across multiple GPUs and nodes.
"""

import os
import torch
import torch.distributed as dist
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DistributedManager:
    """Manager class for distributed training setup and coordination."""

    def __init__(self):
        self.is_initialized = False
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0

    def setup_distributed(
        self, backend: str = "nccl", init_method: Optional[str] = None
    ) -> bool:
        """
        Initialize distributed training environment.

        Args:
            backend: Distributed backend ('nccl', 'gloo', 'mpi')
            init_method: Initialization method URL

        Returns:
            True if successfully initialized, False otherwise
        """
        # Check if we're in a distributed environment
        if not self._is_distributed_environment():
            logger.info("Not in distributed environment, using single GPU/CPU")
            return False

        # Get distributed configuration from environment
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        logger.info(
            f"Initializing distributed training: rank={self.rank}, "
            f"world_size={self.world_size}, local_rank={self.local_rank}"
        )

        try:
            # Initialize the process group
            if not dist.is_initialized():
                if init_method is None:
                    init_method = "env://"

                dist.init_process_group(
                    backend=backend,
                    init_method=init_method,
                    world_size=self.world_size,
                    rank=self.rank,
                )

            # Set the device for this process
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                device = torch.device(f"cuda:{self.local_rank}")
                logger.info(f"Using GPU device: {device}")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU for distributed training")

            self.is_initialized = True
            logger.info("Distributed training initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {e}")
            return False

    def _is_distributed_environment(self) -> bool:
        """Check if we're running in a distributed environment."""
        return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1

    def cleanup(self) -> None:
        """Clean up distributed training resources."""
        if self.is_initialized and dist.is_initialized():
            dist.destroy_process_group()
            self.is_initialized = False
            logger.info("Distributed training cleaned up")

    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.rank == 0

    def barrier(self) -> None:
        """Synchronize all processes."""
        if self.is_initialized:
            dist.barrier()

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensors from all processes."""
        if not self.is_initialized:
            return tensor

        # Create list to hold tensors from all processes
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered_tensors, tensor)

        # Concatenate tensors from all processes
        return torch.cat(gathered_tensors, dim=0)

    def get_world_size(self) -> int:
        """Get the total number of processes."""
        return self.world_size

    def get_rank(self) -> int:
        """Get the rank of this process."""
        return self.rank


def setup_deepspeed_config(
    model_config: Dict[str, Any], training_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate DeepSpeed configuration for large model training.

    Args:
        model_config: Model configuration parameters
        training_config: Training configuration parameters

    Returns:
        DeepSpeed configuration dictionary
    """
    deepspeed_config = {
        "train_batch_size": training_config.get("batch_size", 64),
        "train_micro_batch_size_per_gpu": training_config.get("micro_batch_size", 8),
        "gradient_accumulation_steps": training_config.get(
            "gradient_accumulation_steps", 1
        ),
        # Optimizer configuration
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": training_config.get("learning_rate", 1e-4),
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": training_config.get("weight_decay", 0.01),
            },
        },
        # Learning rate scheduler
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": training_config.get("learning_rate", 1e-4),
                "warmup_num_steps": training_config.get("warmup_steps", 1000),
            },
        },
        # Mixed precision training
        "fp16": {
            "enabled": training_config.get("fp16", True),
            "auto_cast": False,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        # Zero optimization
        "zero_optimization": {
            "stage": 2,  # Stage 2 is usually sufficient for most models
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },
        # Gradient clipping
        "gradient_clipping": training_config.get("gradient_clip_val", 1.0),
        # Wall clock breakdown
        "wall_clock_breakdown": False,
    }

    # Add ZeRO-3 for very large models if specified
    if model_config.get("use_zero3", False):
        deepspeed_config["zero_optimization"]["stage"] = 3
        deepspeed_config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }
        deepspeed_config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    return deepspeed_config


def setup_fsdp_config(
    model_config: Dict[str, Any], training_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate FSDP (Fully Sharded Data Parallel) configuration.

    Args:
        model_config: Model configuration parameters
        training_config: Training configuration parameters

    Returns:
        FSDP configuration dictionary
    """
    fsdp_config = {
        "fsdp_transformer_layer_cls_to_wrap": model_config.get(
            "transformer_layer_cls", ["TransformerBlock", "Block"]
        ),
        "fsdp_backward_prefetch": "backward_pre",
        "fsdp_forward_prefetch": False,
        "fsdp_use_orig_params": True,
        "fsdp_cpu_ram_efficient_loading": True,
    }

    # Add CPU offloading for large models
    if model_config.get("use_cpu_offload", False):
        fsdp_config["fsdp_offload_params"] = True
        fsdp_config["fsdp_state_dict_type"] = "SHARDED_STATE_DICT"

    return fsdp_config


class GPUManager:
    """Utility class for GPU management and optimization."""

    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """Get information about available GPUs."""
        if not torch.cuda.is_available():
            return {"gpu_available": False, "gpu_count": 0}

        gpu_count = torch.cuda.device_count()
        gpu_info = {
            "gpu_available": True,
            "gpu_count": gpu_count,
            "current_device": torch.cuda.current_device(),
            "devices": [],
        }

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            device_info = {
                "id": i,
                "name": props.name,
                "total_memory": props.total_memory,
                "major": props.major,
                "minor": props.minor,
                "multi_processor_count": props.multi_processor_count,
            }
            gpu_info["devices"].append(device_info)

        return gpu_info

    @staticmethod
    def optimize_memory_usage():
        """Apply GPU memory optimizations."""
        if torch.cuda.is_available():
            # Enable memory fraction optimization
            torch.cuda.empty_cache()

            # Set memory growth (if using TensorFlow-like behavior)
            # This is more relevant for other frameworks, but kept for compatibility
            pass

    @staticmethod
    def get_optimal_batch_size(
        model: torch.nn.Module,
        input_shape: tuple,
        device: torch.device,
        max_memory_fraction: float = 0.8,
    ) -> int:
        """
        Estimate optimal batch size based on available GPU memory.

        Args:
            model: PyTorch model
            input_shape: Shape of input data (without batch dimension)
            device: Target device
            max_memory_fraction: Maximum fraction of GPU memory to use

        Returns:
            Estimated optimal batch size
        """
        if not torch.cuda.is_available() or device.type != "cuda":
            return 32  # Default for CPU

        # Get available memory
        total_memory = torch.cuda.get_device_properties(device.index).total_memory
        available_memory = total_memory * max_memory_fraction

        # Estimate memory per sample (rough approximation)
        # This is a simplified calculation - real implementation would be more complex
        model_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        input_memory = torch.tensor(input_shape).prod().item() * 4  # Assume float32

        # Very rough estimate - actual memory usage is more complex
        memory_per_sample = (
            model_memory / 100 + input_memory * 2
        )  # Forward + backward pass

        optimal_batch_size = int(available_memory // memory_per_sample)

        # Clamp to reasonable range
        return max(1, min(optimal_batch_size, 512))


# Global distributed manager instance
distributed_manager = DistributedManager()
