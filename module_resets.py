import importlib


def reset_jax(gallery_conf, fname):
    importlib.reload("jax")

__all__ = ["reset_jax"]
