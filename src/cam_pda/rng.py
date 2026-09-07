"""Local solver RNG policy, preserving the caller state."""
from contextlib import contextmanager
import torch
@contextmanager
def alignment_rng(device, seed):
    """Local official rand/lstsq RNG, restored on success AND failure.

    CPU default lstsq can have nondeterministic workspace last-bit differences;
    use a temporary deterministic policy for CPU only, without another driver.
    CUDA retains the caller's numerical policy and seeds only its used device.
    """
    device = torch.device(device)
    indices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    with torch.random.fork_rng(devices=indices, enabled=True):
        if device.type == 'cuda':
            torch.cuda.default_generators[indices[0]].manual_seed(seed)
        else:
            torch.default_generator.manual_seed(seed)
            torch.use_deterministic_algorithms(True)
        try:
            yield
        finally:
            if device.type == 'cpu':
                torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
