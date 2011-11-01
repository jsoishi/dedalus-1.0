from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
# import distribute_setup
# distribute_setup.use_setuptools()
# import setuptools

def find_fftw():
    import glob
    import os
    path = "/usr/lib"
    l = glob.glob(os.path.join(path,"*fftw*"))
    if len(l) != 0:
        return path
    path = os.path.expanduser("~/build/lib")
    l = glob.glob(os.path.join(path,"*fftw*"))
    if len(l) != 0:
        return path

setup(
    name='dedalus',
    version='0.1dev',
    author='J. S. Oishi',
    author_email='jsoishi@gmail.com',
    license='GPL3',
    packages = ['dedalus.analysis',
                'dedalus.data_objects',
                'dedalus',
                'dedalus.init_cond',
                'dedalus.physics',
                'dedalus.time_stepping',
                'dedalus.utils'],
    include_dirs = [np.get_include()],
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("dedalus.utils.fftw.fftw",
                             ["dedalus/utils/fftw/_fftw.pyx"],
                             libraries=["fftw3"],
                             include_dirs=["dedalus/utils/fftw"],
                             library_dirs=[find_fftw()])]
    
    )
