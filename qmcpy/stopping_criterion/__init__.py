from .abstract_stopping_criterion import AbstractStoppingCriterion
from .cub_mc_clt import CubMCCLT
from .cub_qmc_rep_student_t import CubQMCRepStudentT
from .cub_mc_g import CubMCG
from .cub_qmc_lattice_g import CubQMCLatticeG
from .cub_qmc_net_g import CubQMCNetG
from .cub_mc_ml import CubMCML
from .cub_qmc_ml import CubQMCML
from .cub_mc_ml_cont import CubMCMLCont
from .cub_qmc_ml_cont import CubQMCMLCont
from .cub_qmc_bayes_lattice_g import CubQMCBayesLatticeG
from .cub_qmc_bayes_net_g import CubQMCBayesNetG
from .cub_mc_clt_vec import CubMCCLTVec
try: 
    import torch 
    import gpytorch 
    from .pf_gp_ci import PFGPCI,PFSampleErrorDensityAR,SuggesterSimple
except ImportError:
    class PFGPCI(object):
        def __init__(self, *args, **kwargs):
            raise Exception("PFGPCI requires torch and gpytorch but no installations found")
    class PFSampleErrorDensityAR(object):
        def __init__(self, *args, **kwargs):
            raise Exception("PFSampleErrorDensityAR requires torch and gpytorch but no installations found")
    class SuggesterSimple(object):
        def __init__(self, *args, **kwargs):
            raise Exception("SuggesterSimple requires torch and gpytorch but no installations found")

StoppingCriterion = AbstractStoppingCriterion
_StoppingCriterion = AbstractStoppingCriterion
CubQMCRep = CubQMCRepStudentT
CubQMCCLT = CubQMCRepStudentT
CubQMCDigitalNetB2G = CubQMCNetG
CubQMCSobolG = CubQMCNetG
CubBayesLatticeG = CubQMCBayesLatticeG
CubBayesNetG = CubQMCBayesNetG
CubQMCCBayesDigitalNetB2G = CubQMCBayesNetG
CubBayesDigitalNetB2G = CubQMCBayesNetG
CubQMCBayesSobolG = CubQMCBayesNetG
CubBayesSobolG = CubQMCBayesNetG