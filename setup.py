from distutils.core import setup
from distutils.extension import Extension
from distutils.command.build_py import build_py
from Cython.Distutils import build_ext
import numpy as np
import os
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

def get_mercurial_changeset_id(targetDir):
    """adapted from a script by Jason F. Harris, published at

    http://jasonfharris.com/blog/2010/05/versioning-your-application-with-the-mercurial-changeset-hash/

    """
    import subprocess
    import re
    getChangeset = subprocess.Popen('hg parent --template "{node|short}" --cwd ' + targetDir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        
    if (getChangeset.stderr.read() != ""):
        print "Error in obtaining current changeset of the Mercurial repository"
        changeset = None
        
    changeset = getChangeset.stdout.read()
    if (not re.search("^[0-9a-f]{12}$", changeset)):
        print "Current changeset of the Mercurial repository is malformed"
        changeset = None

    return changeset


class my_build_py(build_py):
    def run(self):
        # honor the --dry-run flag
        if not self.dry_run:
            target_dir = os.path.join(self.build_lib,'dedalus')
            src_dir =  os.getcwd() 
            changeset = get_mercurial_changeset_id(src_dir)
            self.mkpath(target_dir)
            with open(os.path.join(target_dir, '__hg_version__.py'), 'w') as fobj:
                fobj.write("hg_version = '%s'\n" % changeset)

            build_py.run(self)

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
                'dedalus.utils',
                'dedalus.utils.fftw'],
    include_dirs = [np.get_include()],
    cmdclass = {'build_ext': build_ext,
                'build_py': my_build_py},
    ext_modules = [Extension("dedalus.utils.fftw.fftw",
                             ["dedalus/utils/fftw/_fftw.pyx"],
                             libraries=["fftw3"],
                             include_dirs=["dedalus/utils/fftw"],
                             library_dirs=[find_fftw()])]
    
    )
