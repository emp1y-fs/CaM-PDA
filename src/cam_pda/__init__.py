"""CaM-PDA's public RGB-D inference interface."""
__version__ = "0.1.0"

def __getattr__(name):
    if name == "CaMPDA":
        from .pipeline import CaMPDA
        return CaMPDA
    if name == "CameraIntrinsics":
        from .io import CameraIntrinsics
        return CameraIntrinsics
    raise AttributeError(name)

__all__ = ["CaMPDA", "CameraIntrinsics"]
