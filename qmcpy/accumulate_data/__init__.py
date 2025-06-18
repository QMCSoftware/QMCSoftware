from ._accumulate_data import AccumulateData
try: 
    import torch 
    import gpytorch 
    from .pf_gp_ci_data import PFGPCIData
except ImportError:
    class PFGPCIData(object):
        def __init__(self, *args, **kwargs):
            raise Exception("PFGPCIData requires torch and gpytorch but no installations found")


