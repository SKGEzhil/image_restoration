"""Model registry — import and register model factories here."""

from .nafnet import create_model_nafnet
from .scunet import create_model_scunet, create_model_scunet_sr
from .discriminator import create_discriminator

MODEL_REGISTRY = {
    "nafnet": create_model_nafnet,
    "scunet": create_model_scunet,
    "scunet_sr": create_model_scunet_sr,
}


def create_model(name, **kwargs):
    """Instantiate a model by name.

    Args:
        name: key in MODEL_REGISTRY (e.g. "nafnet", "scunet", "scunet_sr")
        **kwargs: model-specific hyperparameters
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
