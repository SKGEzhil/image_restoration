"""Model registry — import and register model factories here."""

from .nafnet import create_model_nafnet

MODEL_REGISTRY = {
    "nafnet": create_model_nafnet,
}


def create_model(name, **kwargs):
    """Instantiate a model by name.

    Args:
        name: key in MODEL_REGISTRY (e.g. "nafnet")
        **kwargs: model-specific hyperparameters
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
