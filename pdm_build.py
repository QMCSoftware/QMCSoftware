from setuptools import Extension

ext_modules = [
        Extension(
            name = 'qmcpy.discrete_distribution._c_lib._c_lib',
            sources = [
                'qmcpy/discrete_distribution/_c_lib/util.c',
                'qmcpy/discrete_distribution/_c_lib/halton_qrng.c',
            ]
        )
    ]

def pdm_build_update_setup_kwargs(context, setup_kwargs):
    setup_kwargs.update(ext_modules=ext_modules)