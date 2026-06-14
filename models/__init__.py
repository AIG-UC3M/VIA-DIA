"""
Model registry for DIA project.

Available architectures:
- BaselineMultimodalModel: Simple CNN+MLP fusion (fast, no attention)
"""

from .baseline_model import BaselineMultimodalModel, create_baseline_model

__all__ = [
    'BaselineMultimodalModel',
    'create_baseline_model',
]


def create_model(arch_name, **kwargs):
    """
    Factory function to create models by name.
    
    Parameters
    ----------
    arch_name : str
        One of: 'BaselineMultimodalModel'
    **kwargs : dict
        Model-specific parameters
        
    Returns
    -------
    model : nn.Module
    """
    if arch_name == 'BaselineMultimodalModel':
        return create_baseline_model(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")
