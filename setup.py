import sys
if sys.version_info < (3,):
    print("Python 2 has reached end-of-life and is no longer supported by PyTorch.")
    sys.exit(-1)
print(f'sys.platform: {sys.platform}')
if not (sys.platform == 'linux' or sys.platform == 'darwin') or sys.maxsize.bit_length() == 31:
    print("32-bit Python runtime is not supported. Please switch to 64-bit Python.")
    print("only supported in linux like 64-bit Platform")
    sys.exit(-1)

import platform
python_min_version = (3, 7, 0)
python_min_version_str = '.'.join(map(str, python_min_version))
if sys.version_info < python_min_version:
    print("You are using Python {}. Python >={} is required.".format(platform.python_version(),
                                                                     python_min_version_str))
    sys.exit(-1)

from setuptools import find_packages, setup, Extension
from setuptools.dist import Distribution

import setuptools.command.build_ext
import setuptools.command.install
import setuptools.command.sdist

from collections import defaultdict

import os
import sysconfig
import json
import glob
import importlib
import shutil

from tools.build_pytorch_libs import build_caffe2
from tools.setup_helpers.env import (BUILD_DIR, IS_LINUX, build_type)
from tools.setup_helpers.cmake import CMake
from tools.generate_torch_version import get_torch_version

VERBOSE_SCRIPT = True
RUN_BUILD_DEPS = True
EMIT_BUILD_WARNING = False
RERUN_CMAKE = False
CMAKE_ONLY = False
filtered_args = []

for i, arg in enumerate(sys.argv):
    if arg == '--cmake':
        RERUN_CMAKE = True
        continue
    if arg == '--cmake-only':
        # Stop once cmake terminates. Leave users a chance to adjust build
        # options.
        CMAKE_ONLY = True
        continue

    if arg == 'rebuild' or arg == 'build':
        arg = 'build' # rebuild is gone, make it build
        EMIT_BUILD_WARNING = True
    if arg == "--":
        filtered_args += sys.argv[i:]
        break
    if arg == '-q' or arg == '--quiet':
        VERBOSE_SCRIPT = False
    if arg in ['clean', 'egg-info', 'sdist']:
        RUN_BUILD_DEPS = False
    filtered_args.append(arg)

sys.argv = filtered_args

if VERBOSE_SCRIPT:
    print('verbose')
    def report(*args):
        print(*args)
else:
    def report(*args):
        pass

cwd = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(cwd, "torch", "lib")
third_party_path = os.path.join(cwd, "third_party")
caffe2_build_dir = os.path.join(cwd, "build")

cmake_python_include_dir = sysconfig.get_path("include")
cmake_python_library = "{}/{}".format(
    sysconfig.get_config_var("LIBDIR"),
    sysconfig.get_config_var("INSTSONAME"))

print(f'cmake_python_include_dir: {cmake_python_include_dir}')
print(f'cmake_python_library: {cmake_python_library}')
print(f'lib_path: {lib_path}')
print(f'third_party_path: {third_party_path}')
print(f'caffe2_build_dir: {caffe2_build_dir}')

main_libraries =  ['torch']
library_dirs = ['torch']
library_dirs.append('./torch/csrc')

################################################################################
# Version, create_version_file, and package_name
################################################################################
package_name = os.getenv('TORCH_PACKAGE_NAME', 'torch')
version = get_torch_version()
report("Building wheel {}-{}".format(package_name, version))

cmake = CMake()
report(f'BUILD_DIR: {BUILD_DIR}')

def get_submodule_folders():
    git_modules_path = os.path.join(cwd, ".gitmodules")
    default_modules_path = [os.path.join(third_party_path, name) for name in [
                            "googletest"
                            # "gloo", "cpuinfo", "tbb", "onnx",
                            # "foxi", "QNNPACK", "fbgemm"
                            ]]
    if not os.path.exists(git_modules_path):
        return default_modules_path
    with open(git_modules_path) as f:
        return [os.path.join(cwd, line.split("=", 1)[1].strip()) for line in
                f.readlines() if line.strip().startswith("path")]

print(f'submodule_folders: {get_submodule_folders()}')

def check_submodules():
    def check_for_files(folder, files):
        if not any(os.path.exists(os.path.join(folder, f)) for f in files):
            report("Could not find any of {} in {}".format(", ".join(files), folder))
            report("Did you run 'git submodule update --init --recursive --jobs 0'?")
            sys.exit(1)

    def not_exists_or_empty(folder):
        return not os.path.exists(folder) or (os.path.isdir(folder) and len(os.listdir(folder)) == 0)

    if bool(os.getenv("USE_SYSTEM_LIBS", False)):
        return
    folders = get_submodule_folders()
    # If none of the submodule folders exists, try to initialize them
    if all(not_exists_or_empty(folder) for folder in folders):
        try:
            print(' --- Trying to initialize submodules')
            start = time.time()
            subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=cwd)
            end = time.time()
            print(' --- Submodule initialization took {:.2f} sec'.format(end - start))
        except Exception:
            print(' --- Submodule initalization failed')
            print('Please run:\n\tgit submodule update --init --recursive --jobs 0')
            sys.exit(1)
    for folder in folders:
        check_for_files(folder, ["CMakeLists.txt", "Makefile", "setup.py", "LICENSE", "LICENSE.md", "LICENSE.txt"])
    check_for_files(os.path.join(third_party_path, 'fbgemm', 'third_party',
                                 'asmjit'), ['CMakeLists.txt'])
    # check_for_files(os.path.join(third_party_path, 'onnx', 'third_party',
    #                              'benchmark'), ['CMakeLists.txt'])


install_requires = [
    # "typing_extensions",
]

class build_ext(setuptools.command.build_ext.build_ext):

    def run(self):
        # Report build options. This is run after the build completes so # `CMakeCache.txt` exists and we can get an
        # accurate report on what is used and what is not.
        cmake_cache_vars = defaultdict(lambda: False, cmake.get_cmake_cache_variables())

        if cmake_cache_vars['USE_CUDA']:
            report('-- Detected CUDA at ' + cmake_cache_vars['CUDA_TOOLKIT_ROOT_DIR'])
        else:
            report('-- Not using CUDA')

        if cmake_cache_vars['USE_NCCL'] and cmake_cache_vars['USE_SYSTEM_NCCL']:
            report('-- Using system provided NCCL library at {}, {}'.format(cmake_cache_vars['NCCL_LIBRARIES'],
                                                                            cmake_cache_vars['NCCL_INCLUDE_DIRS']))
        elif cmake_cache_vars['USE_NCCL']:
            report('-- Building NCCL library')
        else:
            report('-- Not using NCCL')
        if cmake_cache_vars['USE_DISTRIBUTED']:
            report('-- Building with distributed package: ')
            report('  -- USE_TENSORPIPE={}'.format(cmake_cache_vars['USE_TENSORPIPE']))
            report('  -- USE_GLOO={}'.format(cmake_cache_vars['USE_GLOO']))
            report('  -- USE_MPI={}'.format(cmake_cache_vars['USE_OPENMPI']))
        else:
            report('-- Building without distributed package')

        # if cmake_cache_vars['STATIC_DISPATCH_BACKEND']:
        #     report('-- Using static dispatch with backend {}'.format(cmake_cache_vars['STATIC_DISPATCH_BACKEND']))
        # if cmake_cache_vars['USE_LIGHTWEIGHT_DISPATCH']:
        #     report('-- Using lightweight dispatch')

        # Do not use clang to compile extensions if `-fstack-clash-protection` is defined
        # in system CFLAGS
        c_flags = str(os.getenv('CFLAGS', ''))
        if IS_LINUX and '-fstack-clash-protection' in c_flags and 'clang' in os.environ.get('CC', ''):
            os.environ['CC'] = str(os.environ['CC'])

        # It's an old-style class in Python 2.7...
        setuptools.command.build_ext.build_ext.run(self)


    def build_extensions(self):
        self.create_compile_commands()
        # The caffe2 extensions are created in
        # tmp_install/lib/pythonM.m/site-packages/caffe2/python/
        # and need to be copied to build/lib.linux.... , which will be a
        # platform dependent build folder created by the "build" command of
        # setuptools. Only the contents of this folder are installed in the
        # "install" command by default.
        # We only make this copy for Caffe2's pybind extensions
        caffe2_pybind_exts = [
            # 'caffe2.python.caffe2_pybind11_state',
            # 'caffe2.python.caffe2_pybind11_state_gpu',
        ]
        i = 0
        while i < len(self.extensions):
            ext = self.extensions[i]
            if ext.name not in caffe2_pybind_exts:
                i += 1
                continue
            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)
            report("\nCopying extension {}".format(ext.name))

            relative_site_packages = sysconfig.get_path('purelib').replace(sysconfig.get_path('data'), '').lstrip(os.path.sep)
            src = os.path.join("torch", relative_site_packages, filename)
            if not os.path.exists(src):
                report("{} does not exist".format(src))
                del self.extensions[i]
            else:
                dst = os.path.join(os.path.realpath(self.build_lib), filename)
                report("Copying {} from {} to {}".format(ext.name, src, dst))
                dst_dir = os.path.dirname(dst)
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                self.copy_file(src, dst)
                i += 1
        setuptools.command.build_ext.build_ext.build_extensions(self)


    def get_outputs(self):
        outputs = setuptools.command.build_ext.build_ext.get_outputs(self)
        outputs.append(os.path.join(self.build_lib, "caffe2"))
        report("setup.py::get_outputs returning {}".format(outputs))
        return outputs

    def create_compile_commands(self):
        def load(filename):
            with open(filename) as f:
                return json.load(f)
        ninja_files = glob.glob('build/*compile_commands.json')
        cmake_files = glob.glob('torch/lib/build/*/compile_commands.json')
        all_commands = [entry
                        for f in ninja_files + cmake_files
                        for entry in load(f)]

        # cquery does not like c++ compiles that start with gcc.
        # It forgets to include the c++ header directories.
        # We can work around this by replacing the gcc calls that python
        # setup.py generates with g++ calls instead
        for command in all_commands:
            if command['command'].startswith("gcc "):
                command['command'] = "g++ " + command['command'][4:]

        new_contents = json.dumps(all_commands, indent=2)
        contents = ''
        if os.path.exists('compile_commands.json'):
            with open('compile_commands.json', 'r') as f:
                contents = f.read()
        if contents != new_contents:
            with open('compile_commands.json', 'w') as f:
                f.write(new_contents)

class concat_license_files():
    """Merge LICENSE and LICENSES_BUNDLED.txt as a context manager

    LICENSE is the main PyTorch license, LICENSES_BUNDLED.txt is auto-generated
    from all the licenses found in ./third_party/. We concatenate them so there
    is a single license file in the sdist and wheels with all of the necessary
    licensing info.
    """
    def __init__(self):
        self.f1 = 'LICENSE'
        self.f2 = 'third_party/LICENSES_BUNDLED.txt'

    def __enter__(self):
        """Concatenate files"""
        with open(self.f1, 'r') as f1:
            self.bsd_text = f1.read()

        with open(self.f1, 'a') as f1:
            with open(self.f2, 'r') as f2:
                self.bundled_text = f2.read()
                f1.write('\n\n')
                f1.write(self.bundled_text)

    def __exit__(self, exception_type, exception_value, traceback):
        """Restore content of f1"""
        with open(self.f1, 'w') as f:
            f.write(self.bsd_text)

try:
    from wheel.bdist_wheel import bdist_wheel
except ImportError:
    # This is useful when wheel is not installed and bdist_wheel is not
    # specified on the command line. If it _is_ specified, parsing the command
    # line will fail before wheel_concatenate is needed
    wheel_concatenate = None
else:
    # Need to create the proper LICENSE.txt for the wheel
    class wheel_concatenate(bdist_wheel):
        """ check submodules on sdist to prevent incomplete tarballs """
        def run(self):
            with concat_license_files():
                super().run()

class install(setuptools.command.install.install):
    def run(self):
        super().run()


class clean(setuptools.Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import glob
        import re
        with open('.gitignore', 'r') as f:
            ignores = f.read()
            pat = re.compile(r'^#( BEGIN NOT-CLEAN-FILES )?')
            for wildcard in filter(None, ignores.split('\n')):
                if wildcard == '.vscode':
                    # do not remove .vscode
                    continue
                match = pat.match(wildcard)
                if match:
                    if match.group(1):
                        # Marker is found and stop reading .gitignore.
                        break
                    # Ignore lines which begin with '#'.
                else:
                    for filename in glob.glob(wildcard):
                        try:
                            os.remove(filename)
                        except OSError:
                            shutil.rmtree(filename, ignore_errors=True)


class sdist(setuptools.command.sdist.sdist):
    def run(self):
        with concat_license_files():
            super().run()

def configure_extension_build():
    r"""Configures extension build options according to system environment and user's choice.

    Returns:
      The input to parameters ext_modules, cmdclass, packages, and entry_points as required in setuptools.setup.
    """
    try:
        cmake_cache_vars = defaultdict(lambda: False, cmake.get_cmake_cache_variables())
    except FileNotFoundError:
        # CMakeCache.txt does not exist. Probably running "python setup.py clean" over a clean directory.
        cmake_cache_vars = defaultdict(lambda: False)

    ################################################################################
    # Configure compile flags
    ################################################################################

    library_dirs = []
    extra_install_requires = []
    extra_link_args = []
    extra_compile_args = [
                '-Wall',
                '-Wextra',
                '-Wno-strict-overflow',
                '-Wno-unused-parameter',
                '-Wno-missing-field-initializers',
                '-Wno-write-strings',
                '-Wno-unknown-pragmas',
                # This is required for Python 2 declarations that are deprecated in 3.
                '-Wno-deprecated-declarations',
                # Python 2.6 requires -fno-strict-aliasing, see
                # http://legacy.python.org/dev/peps/pep-3123/
                # We also depend on it in our code (even Python 3).
                '-fno-strict-aliasing',
                # Clang has an unfixed bug leading to spurious missing
                # braces warnings, see
                # https://bugs.llvm.org/show_bug.cgi?id=21629
                '-Wno-missing-braces',
            ]

    library_dirs.append(lib_path)

    main_compile_args = []
    main_libraries = ['torch_python']
    main_link_args = []
    main_sources = ["torch/csrc/stub.c"]
    if cmake_cache_vars['USE_CUDA']:
        library_dirs.append(
            os.path.dirname(cmake_cache_vars['CUDA_CUDA_LIB']))


    if build_type.is_debug():
        extra_compile_args += ['-O0', '-g']
        extra_link_args += ['-O0', '-g']

    if build_type.is_rel_with_deb_info():
        extra_compile_args += ['-g']
        extra_link_args += ['-g']

    def make_relative_rpath_args(path):
        return ['-Wl,-rpath,$ORIGIN/' + path]


    ################################################################################
    # Declare extensions and package
    ################################################################################

    extensions = []
    packages = find_packages(exclude=('tools', 'tools.*'))
    print(f'packages: {packages}')

    C = Extension("torch._C",
                  libraries=main_libraries,
                  sources=main_sources,
                  language='c',
                  extra_compile_args=main_compile_args + extra_compile_args,
                  include_dirs=[],
                  library_dirs=library_dirs,
                  extra_link_args=extra_link_args + main_link_args + make_relative_rpath_args('lib'))
    # C_flatbuffer = Extension("torch._C_flatbuffer",
    #                          libraries=main_libraries,
    #                          sources=["torch/csrc/stub_with_flatbuffer.c"],
    #                          language='c',
    #                          extra_compile_args=main_compile_args + extra_compile_args,
    #                          include_dirs=[],
    #                          library_dirs=library_dirs,
    #                          extra_link_args=extra_link_args + main_link_args + make_relative_rpath_args('lib'))
    extensions.append(C)
    # extensions.append(C_flatbuffer)

    # DL = Extension("torch._dl",
    #                sources=["torch/csrc/dl.c"],
    #                language='c')
    # extensions.append(DL)


    cmdclass = {
        'bdist_wheel': wheel_concatenate,
        'build_ext': build_ext,
        'clean': clean,
        'install': install,
        'sdist': sdist,
    }

    entry_points = {
        'console_scripts': [
            # 'convert-caffe2-to-onnx = caffe2.python.onnx.bin.conversion:caffe2_to_onnx',
            # 'convert-onnx-to-caffe2 = caffe2.python.onnx.bin.conversion:onnx_to_caffe2',
            # 'torchrun = torch.distributed.run:main',
        ]
    }
    return extensions, cmdclass, packages, entry_points, extra_install_requires

def check_pydep(importname, module):
    try:
        importlib.import_module(importname)
    except ImportError:
        raise RuntimeError(missing_pydep.format(importname=importname, module=module))

def build_deps():
    report('-- Building version ' + version)
    check_submodules()
    # check_pydep('yaml', 'pyyaml')
    build_caffe2(version=version,
                 cmake_python_library=cmake_python_library,
                 build_python=True,
                 rerun_cmake=RERUN_CMAKE,
                 cmake_only=CMAKE_ONLY,
                 cmake=cmake)

    if CMAKE_ONLY:
        report('Finished running cmake. Run "ccmake build" or '
               '"cmake-gui build" to adjust build options and '
               '"python setup.py install" to build.')
        sys.exit()


# post run, warnings, printed at the end to make them more visible
build_update_message = """
    It is no longer necessary to use the 'build' or 'rebuild' targets

    To install:
      $ python setup.py install
    To develop locally:
      $ python setup.py develop
    To force cmake to re-generate native build files (off by default):
      $ python setup.py develop --cmake
"""

def print_box(msg):
    lines = msg.split('\n')
    size = max(len(l) + 1 for l in lines)
    print('-' * (size + 2))
    for l in lines:
        print('|{}{}|'.format(l, ' ' * (size - len(l))))
    print('-' * (size + 2))

if __name__ == '__main__':
    dist = Distribution()
    try:
        dist.parse_command_line()
    except setuptools.distutils.errors.DistutilsArgError as e:
        print(e)
        sys.exit(1)

    if RUN_BUILD_DEPS:
        build_deps()

    extensions, cmdclass, packages, entry_points, extra_install_requires = configure_extension_build()

    print(f"packages: {packages}")

    install_requires += extra_install_requires

    # Read in README.md for our long_description
    with open(os.path.join(cwd, "README.md"), encoding='utf-8') as f:
        long_description = f.read()

    version_range_max = max(sys.version_info[1], 9) + 1

    print(f'version_range_max: {version_range_max}')
    print(f'sys.version_info: {sys.version_info}')

    setup(
        name = package_name,
        version = version,
        description = ("Tensors and Dynamic neural networks in "
                     "Python with strong GPU acceleration"),
        long_description = long_description,
        long_description_content_type = "text/markdown",
        ext_modules = extensions,
        cmdclass = cmdclass,
        packages = packages,
        entry_points = entry_points,
        install_requires = install_requires,
        package_data = {
            'torch': [
                'bin/*',
                'lib/*.so*',
                'lib/*.lib'
            ],
            'caffe2': [],
        },
        url = 'pxcai@abeliancap.com',
        download_url = 'pxcai@abeliancap.com',
        author = 'pxcai',
        author_email = 'pxcai@abeliancap.com',
        python_requires = '>={}'.format(python_min_version_str)
    )

    if EMIT_BUILD_WARNING:
        print_box(build_update_message)

    # module = Extension('torch._C',
    #                 sources = ['torch/csrc/stub.c'],
    #                 libraries = main_libraries,
    #                 library_dirs = library_dirs,
    #                 language = 'c')
    # setup(name = 'pxtorch', version = '1.0', description = 'This is a pxtorch module', ext_modules = [module])
