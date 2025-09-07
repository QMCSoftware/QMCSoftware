"""
GBM demonstration package.

This package provides utilities for Geometric Brownian Motion demonstrations
and maintains backward compatibility for imports.
"""

# Import all utility modules from the code subdirectory and make them 
# available in the current namespace for backward compatibility
import sys
import os

# Add the code directory to the module search path
code_path = os.path.join(os.path.dirname(__file__), 'code')
if code_path not in sys.path:
    sys.path.insert(0, code_path)

# Now import the utility modules so they're available at this level
try:
    from code import config
    from code import data_util  
    from code import latex_util
    from code import plot_util
    from code import qmcpy_util
    from code import quantlib_util
    
    # Make them available as top-level modules in this namespace
    import code.config as config
    import code.data_util as data_util
    import code.latex_util as latex_util  
    import code.plot_util as plot_util
    import code.qmcpy_util as qmcpy_util
    import code.quantlib_util as quantlib_util
    
except ImportError as e:
    print(f"Warning: Could not import utility modules: {e}")
