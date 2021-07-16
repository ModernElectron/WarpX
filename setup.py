import os
import re
import sys
import platform
import shutil
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.command.build import build
from distutils.version import LooseVersion


class CopyPreBuild(build):
    def initialize_options(self):
        build.initialize_options(self)
        # We just overwrite this because the default "build/lib" clashes with
        # directories many developers have in their source trees;
        # this can create confusing results with "pip install .", which clones
        # the whole source tree by default
        self.build_lib = '_tmppythonbuild'

    def run(self):
        build.run(self)

        # matches: libwarpx.(2d|3d|rz).(so|pyd)
        re_libprefix = re.compile(r"libwarpx\...\.(?:so|dll)")
        libs_found = []
        for lib_name in os.listdir(PYWARPX_LIB_DIR):
            if re_libprefix.match(lib_name):
                lib_path = os.path.join(PYWARPX_LIB_DIR, lib_name)
                libs_found.append(lib_path)
        if len(libs_found) == 0:
            raise RuntimeError("Error: no pre-build WarpX libraries found in "
                               "PYWARPX_LIB_DIR='{}'".format(PYWARPX_LIB_DIR))

        # copy external libs into collection of files in a temporary build dir
        dst_path = os.path.join(self.build_lib, "pywarpx")
        for lib_path in libs_found:
            shutil.copy(lib_path, dst_path)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake 3.15.0+ must be installed to build the following " +
                "extensions: " +
                ", ".join(e.name for e in self.extensions))

        cmake_version = LooseVersion(re.search(
            r'version\s*([\d.]+)',
            out.decode()
        ).group(1))
        if cmake_version < '3.15.0':
            raise RuntimeError("CMake >= 3.15.0 is required")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(
            self.get_ext_fullpath(ext.name)
        ))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        r_dim = re.search(r'warpx_(2|3|rz)(?:d*)', ext.name)
        dims = r_dim.group(1).upper()

        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' +
            os.path.join(extdir, "pywarpx"),
            '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=' + extdir,
            '-DWarpX_DIMS=' + dims,
            '-DWarpX_APP:BOOL=OFF',
            '-DWarpX_LIB:BOOL=ON',
            ## variants
            '-DWarpX_COMPUTE=' + WarpX_COMPUTE,
            '-DWarpX_MPI:BOOL=' + WarpX_MPI,
            '-DWarpX_MPI:BOOL=' + WarpX_EB,
            '-DWarpX_OPENPMD:BOOL=' + WarpX_OPENPMD,
            '-DWarpX_PRECISION=' + WarpX_PRECISION,
            '-DWarpX_PSATD:BOOL=' + WarpX_PSATD,
            '-DWarpX_QED:BOOL=' + WarpX_QED,
            '-DWarpX_QED_TABLE_GEN:BOOL=' + WarpX_QED_TABLE_GEN,
            ## dependency control (developers & package managers)
            '-DWarpX_amrex_internal=' + WarpX_amrex_internal,
            #        see PICSAR and openPMD below
            ## static/shared libs
            '-DBUILD_SHARED_LIBS:BOOL=' + BUILD_SHARED_LIBS,
            ## Modern electron variables
            '-DME_Solver:BOOL=' + ME_Solver,
            ## Unix: rpath to current dir when packaged
            ##       needed for shared (here non-default) builds and ADIOS1
            ##       wrapper libraries
            '-DCMAKE_BUILD_WITH_INSTALL_RPATH:BOOL=ON',
            '-DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=OFF',
            # Windows: has no RPath concept, all `.dll`s must be in %PATH%
            #          or same dir as calling executable
        ]
        if WarpX_QED.upper() in ['1', 'ON', 'TRUE', 'YES']:
            cmake_args.append('-DWarpX_picsar_internal=' + WarpX_picsar_internal)
        if WarpX_OPENPMD.upper() in ['1', 'ON', 'TRUE', 'YES']:
            cmake_args += [
                '-DHDF5_USE_STATIC_LIBRARIES:BOOL=' + HDF5_USE_STATIC_LIBRARIES,
                '-DADIOS_USE_STATIC_LIBS:BOOL=' + ADIOS_USE_STATIC_LIBS,
                '-DWarpX_openpmd_internal=' + WarpX_openpmd_internal,
            ]
        # further dependency control (developers & package managers)
        if WarpX_amrex_src:
            cmake_args.append('-DWarpX_amrex_src=' + WarpX_amrex_src)
        if WarpX_openpmd_src:
            cmake_args.append('-DWarpX_openpmd_src=' + WarpX_openpmd_src)
        if WarpX_picsar_src:
            cmake_args.append('-DWarpX_picsar_src=' + WarpX_picsar_src)

        if sys.platform == "darwin":
            cmake_args.append('-DCMAKE_INSTALL_RPATH=@loader_path')
        else:
            # values: linux*, aix, freebsd, ...
            #   just as well win32 & cygwin (although Windows has no RPaths)
            cmake_args.append('-DCMAKE_INSTALL_RPATH=$ORIGIN')

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += [
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                    cfg.upper(),
                    os.path.join(extdir, "pywarpx")
                )
            ]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]

        build_args += ['--parallel', BUILD_PARALLEL]

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version()
        )
        build_dir = os.path.join(self.build_temp, dims)
        os.makedirs(build_dir, exist_ok=True)
        subprocess.check_call(
            ['cmake', ext.sourcedir] + cmake_args,
            cwd=build_dir,
            env=env
        )
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args,
            cwd=build_dir
        )
        # note that this does not call install;
        # we pick up artifacts directly from the build output dirs


with open('./README.md', encoding='utf-8') as f:
    long_description = f.read()

# Allow to control options via environment vars.
#   Work-around for https://github.com/pypa/setuptools/issues/1712
# Pick up existing WarpX libraries or...
PYWARPX_LIB_DIR = os.environ.get('PYWARPX_LIB_DIR')

# ... build WarpX libraries with CMake
#   note: changed default for SHARED, MPI, TESTING and EXAMPLES
WarpX_COMPUTE = os.environ.get('WarpX_COMPUTE', 'OMP')
WarpX_MPI = os.environ.get('WarpX_MPI', 'OFF')
WarpX_EB = os.environ.get('WarpX_EB', 'ON')
WarpX_OPENPMD = os.environ.get('WarpX_OPENPMD', 'OFF')
WarpX_PRECISION = os.environ.get('WarpX_PRECISION', 'DOUBLE')
WarpX_PSATD = os.environ.get('WarpX_PSATD', 'OFF')
WarpX_QED = os.environ.get('WarpX_QED', 'ON')
WarpX_QED_TABLE_GEN = os.environ.get('WarpX_QED_TABLE_GEN', 'OFF')
WarpX_DIMS = os.environ.get('WarpX_DIMS', '2;3;RZ')
BUILD_PARALLEL = os.environ.get('BUILD_PARALLEL', '2')
BUILD_SHARED_LIBS = os.environ.get('WarpX_BUILD_SHARED_LIBS',
                                   'OFF')
# Modern Electron Build Variables
ME_Solver = os.environ.get('ME_Solver', 'OFF')
#BUILD_TESTING = os.environ.get('WarpX_BUILD_TESTING',
#                               'OFF')
#BUILD_EXAMPLES = os.environ.get('WarpX_BUILD_EXAMPLES',
#                                'OFF')
# openPMD-api sub-control
HDF5_USE_STATIC_LIBRARIES = os.environ.get('HDF5_USE_STATIC_LIBRARIES', 'OFF')
ADIOS_USE_STATIC_LIBS = os.environ.get('ADIOS_USE_STATIC_LIBS', 'OFF')
# CMake dependency control (developers & package managers)
WarpX_amrex_src = os.environ.get('WarpX_amrex_src')
WarpX_amrex_internal = os.environ.get('WarpX_amrex_internal', 'ON')
WarpX_openpmd_src = os.environ.get('WarpX_openpmd_src')
WarpX_openpmd_internal = os.environ.get('WarpX_openpmd_internal', 'ON')
WarpX_picsar_src = os.environ.get('WarpX_picsar_src')
WarpX_picsar_internal = os.environ.get('WarpX_picsar_internal', 'ON')

# https://cmake.org/cmake/help/v3.0/command/if.html
if WarpX_MPI.upper() in ['1', 'ON', 'TRUE', 'YES']:
    WarpX_MPI = "ON"
else:
    WarpX_MPI = "OFF"

# Include embedded boundary functionality
if WarpX_EB.upper() in ['1', 'ON', 'TRUE', 'YES']:
    WarpX_EB = "ON"

# Modern Electron variables read environment variables
if ME_Solver.upper() in ['1', 'ON', 'TRUE', 'YES']:
    ME_Solver = "ON"
else:
    ME_Solver = "OFF"

# for CMake
cxx_modules = []     # values: warpx_2d, warpx_3d, warpx_rz
cmdclass = {}        # build extensions

# externally pre-built: pick up pre-built WarpX libraries
if PYWARPX_LIB_DIR:
    cmdclass=dict(build=CopyPreBuild)
# CMake: build WarpX libraries ourselves
else:
    cmdclass = dict(build_ext=CMakeBuild)
    for dim in [x.lower() for x in WarpX_DIMS.split(';')]:
        name = dim if dim == "rz" else dim + "d"
        cxx_modules.append(CMakeExtension("warpx_" + name))

# Get the package requirements from the requirements.txt file
install_requires = []
with open('./requirements.txt') as f:
    install_requires = [line.strip('\n') for line in f.readlines()]
    if WarpX_MPI == "ON":
        install_requires.append('mpi4py>=2.1.0')

# keyword reference:
#   https://packaging.python.org/guides/distributing-packages-using-setuptools
setup(
    name='pywarpx',
    # note PEP-440 syntax: x.y.zaN but x.y.z.devN
    version = '21.07',
    packages = ['pywarpx'],
    package_dir = {'pywarpx': 'Python/pywarpx'},
    author='Jean-Luc Vay, David P. Grote, Maxence Thévenet, Rémi Lehe, Andrew Myers, Weiqun Zhang, Axel Huebl, et al.',
    author_email='jlvay@lbl.gov, grote1@llnl.gov, maxence.thevenet@desy.de, rlehe@lbl.gov, atmyers@lbl.gov, WeiqunZhang@lbl.gov, axelhuebl@lbl.gov',
    maintainer='Axel Huebl, David P. Grote, Rémi Lehe', # wheel/pypi packages
    maintainer_email='axelhuebl@lbl.gov, grote1@llnl.gov, rlehe@lbl.gov',
    description='WarpX is an advanced electromagnetic Particle-In-Cell code.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=('WarpX openscience mpi hpc research pic particle-in-cell '
              'plasma laser-plasma accelerator modeling simulation'),
    url='https://ecp-warpx.github.io',
    project_urls={
        'Documentation': 'https://warpx.readthedocs.io',
        'Doxygen': 'https://warpx.readthedocs.io/en/latest/_static/doxyhtml/index.html',
        #'Reference': 'https://doi.org/...', (Paper and/or Zenodo)
        'Source': 'https://github.com/ECP-WarpX/WarpX',
        'Tracker': 'https://github.com/ECP-WarpX/WarpX/issues',
    },
    # CMake: self-built as extension module
    ext_modules=cxx_modules,
    cmdclass=cmdclass,
    # scripts=['warpx_2d', 'warpx_3d', 'warpx_rz'],
    zip_safe=False,
    python_requires='>=3.6, <3.10',
    # tests_require=['pytest'],
    install_requires=install_requires,
    # see: src/bindings/python/cli
    #entry_points={
    #    'console_scripts': [
    #        'warpx_3d = warpx.3d.__main__:main'
    #    ]
    #},
    extras_require={
        'all': ['openPMD-api~=0.13.0', 'openPMD-viewer~=1.1', 'yt~=3.6', 'matplotlib'],
    },
    # cmdclass={'test': PyTest},
    # platforms='any',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        ('License :: OSI Approved :: '
         'BSD License'), # TODO: use real SPDX: BSD-3-Clause-LBNL
    ],
    # new PEP 639 format
    license='BSD-3-Clause-LBNL',
    license_files = ['LICENSE.txt', 'LEGAL.txt'],
)

