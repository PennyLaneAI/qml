import importlib


def reset_jax(gallery_conf, fname):
	import jax
    importlib.reload(jax)

__all__ = ["reset_jax"]
