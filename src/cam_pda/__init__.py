"""CaM-PDA's public RGB-D inference interface."""
__version__ = "0.4.0"

def __getattr__(name):
    if name == "CaMPDA":
        from .pipeline import CaMPDA
        return CaMPDA
    if name == "CameraIntrinsics":
        from .io import CameraIntrinsics
        return CameraIntrinsics
    if name in {"run_from_paths", "run_example"}:
        from . import runner
        return getattr(runner, name)
    raise AttributeError(name)

__all__ = ["CaMPDA", "CameraIntrinsics", "run_from_paths", "run_example"]
