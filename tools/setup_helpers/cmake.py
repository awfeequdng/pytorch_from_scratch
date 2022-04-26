
import os
import re
from subprocess import check_call, check_output, CalledProcessError
from distutils.version import LooseVersion
from typing import Optional, Union, IO, Dict, Any, List
import multiprocessing

from . import which
from .env import (BUILD_DIR)

def _mkdir_p(d: str) -> None:
    try:
        os.makedirs(d, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Failed to create folder {os.path.abspath(d)}: {e.strerror}") from e

CMakeValue = Optional[Union[bool, str]]


def convert_cmake_value_to_python_value(cmake_value: str, cmake_type: str) -> CMakeValue:
    r"""Convert a CMake value in a string form to a Python value.

    Args:
      cmake_value (string): The CMake value in a string form (e.g., "ON", "OFF", "1").
      cmake_type (string): The CMake type of :attr:`cmake_value`.

    Returns:
      A Python value corresponding to :attr:`cmake_value` with type :attr:`cmake_type`.
    """

    cmake_type = cmake_type.upper()
    up_val = cmake_value.upper()
    if cmake_type == 'BOOL':
        # https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/VariablesListsStrings#boolean-values-in-cmake
        return not (up_val in ('FALSE', 'OFF', 'N', 'NO', '0', '', 'NOTFOUND') or up_val.endswith('-NOTFOUND'))
    elif cmake_type == 'FILEPATH':
        if up_val.endswith('-NOTFOUND'):
            return None
        else:
            return cmake_value
    else:  # Directly return the cmake_value.
        return cmake_value

def get_cmake_cache_variables_from_file(cmake_cache_file: IO[str]) -> Dict[str, CMakeValue]:
    r"""Gets values in CMakeCache.txt into a dictionary.

    Args:
      cmake_cache_file: A CMakeCache.txt file object.
    Returns:
      dict: A ``dict`` containing the value of cached CMake variables.
    """

    results = dict()
    for i, line in enumerate(cmake_cache_file, 1):
        line = line.strip()
        if not line or line.startswith(('#', '//')):
            # Blank or comment line, skip
            continue

        # Almost any character can be part of variable name and value. As a practical matter, we assume the type must be
        # valid if it were a C variable name. It should match the following kinds of strings:
        #
        #   USE_CUDA:BOOL=ON
        #   "USE_CUDA":BOOL=ON
        #   USE_CUDA=ON
        #   USE_CUDA:=ON
        #   Intel(R) MKL-DNN_SOURCE_DIR:STATIC=/path/to/pytorch/third_party/ideep/mkl-dnn
        #   "OpenMP_COMPILE_RESULT_CXX_openmp:experimental":INTERNAL=FALSE
        matched = re.match(r'("?)(.+?)\1(?::\s*([a-zA-Z_-][a-zA-Z0-9_-]*)?)?\s*=\s*(.*)', line)
        if matched is None:  # Illegal line
            raise ValueError('Unexpected line {} in {}: {}'.format(i, repr(cmake_cache_file), line))
        _, variable, type_, value = matched.groups()
        if type_ is None:
            type_ = ''
        if type_.upper() in ('INTERNAL', 'STATIC'):
            # CMake internal variable, do not touch
            continue
        results[variable] = convert_cmake_value_to_python_value(value, type_)

    return results

class CMake:
    def __init__(self, build_dir: str = BUILD_DIR) -> None:
        self._cmake_command = CMake._get_cmake_command()
        self.build_dir = build_dir

    @property
    def _cmake_cache_file(self) -> str:
        r"""Returns the path to CMakeCache.txt.

        Returns:
          string: The path to CMakeCache.txt.
        """
        return os.path.join(self.build_dir, 'CMakeCache.txt')

    @staticmethod
    def _get_cmake_command() -> str:
        "Returns cmake command."

        cmake_command = 'cmake'
        cmake3_version = CMake._get_version(which('cmake3'))
        cmake_version = CMake._get_version(which('cmake'))

        _cmake_min_version = LooseVersion("3.10.0")
        if all((ver is None or ver < _cmake_min_version for ver in [cmake_version, cmake3_version])):
            raise RuntimeError('no cmake or cmake3 with version >= 3.10.0 found')

        if cmake3_version is None:
            cmake_command = 'cmake'
        elif cmake_version is None:
            cmake_command = 'cmake3'
        else:
            if cmake3_version >= cmake_version:
                cmake_command = 'cmake3'
            else:
                cmake_command = 'cmake'
        return cmake_command

    @staticmethod
    def _get_version(cmd: Optional[str]) -> Any:
        "Returns cmake version."

        if cmd is None:
            return None
        for line in check_output([cmd, '--version']).decode('utf-8').split('\n'):
            if 'version' in line:
                return LooseVersion(line.strip().split(' ')[2])
        raise RuntimeError('no version found')

    def run(self, args: List[str], env: Dict[str, str]) -> None:
        command = [self._cmake_command] + args
        print(' '.join(command))
        try:
            check_call(command, cwd=self.build_dir, env=env)
        except (CalledProcessError, KeyboardInterrupt) as e:
            sys.exit(1)

    @staticmethod
    def defines(args: List[str], **kwargs: CMakeValue) -> None:
        for key, value in sorted(kwargs.items()):
            if value is not None:
                args.append('-D{}={}'.format(key, value))

    def get_cmake_cache_variables(self) -> Dict[str, CMakeValue]:
        with open(self._cmake_cache_file) as f:
            return get_cmake_cache_variables_from_file(f)

    def generate(
        self,
        version: Optional[str],
        cmake_python_library: Optional[str],
        build_python: bool,
        build_test: bool,
        my_env: Dict[str, str],
        rerun: bool,
    ) -> None:
        if rerun and os.path.isfile(self._cmake_cache_file):
            os.remove(self._cmake_cache_file)
        print(f'self._cmake_cache_file: {self._cmake_cache_file}')
        if os.path.exists(self._cmake_cache_file):
            # Everything's in place. Do not rerun.
            return

        args = []

        print(f'__file__: {__file__}')

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        install_dir = os.path.join(base_dir, "torch")
        print(f'install_dir: {install_dir}')
        _mkdir_p(install_dir)
        _mkdir_p(self.build_dir)

        # Store build options that are directly stored in environment variables
        build_options: Dict[str, CMakeValue] = {}

        # Build options that do not start with "BUILD_", "USE_", or "CMAKE_" and are directly controlled by env vars.
        # This is a dict that maps environment variables to the corresponding variable name in CMake.
        additional_options = {
            # Key: environment variable name. Value: Corresponding variable name to be passed to CMake. If you are
            # adding a new build option to this block: Consider making these two names identical and adding this option
            # in the block below.
        }
        additional_options.update({
            # Build options that have the same environment variable name and CMake variable name and that do not start
            # with "BUILD_", "USE_", or "CMAKE_". If you are adding a new build option, also make sure you add it to
            # CMakeLists.txt.
        })

        # Aliases which are lower priority than their canonical option
        low_priority_aliases = {
        }

        for var, val in my_env.items():
            # We currently pass over all environment variables that start with "BUILD_", "USE_", and "CMAKE_". This is
            # because we currently have no reliable way to get the list of all build options we have specified in
            # CMakeLists.txt. (`cmake -L` won't print dependent options when the dependency condition is not met.) We
            # will possibly change this in the future by parsing CMakeLists.txt ourselves (then additional_options would
            # also not be needed to be specified here).
            true_var = additional_options.get(var)
            if true_var is not None:
                build_options[true_var] = val
            elif var.startswith(('BUILD_', 'USE_', 'CMAKE_')) or var.endswith(('EXITCODE', 'EXITCODE__TRYRUN_OUTPUT')):
                build_options[var] = val

            if var in low_priority_aliases:
                key = low_priority_aliases[var]
                if key not in build_options:
                    build_options[key] = val

        # The default value cannot be easily obtained in CMakeLists.txt. We set it here.
        py_lib_path = sysconfig.get_path('purelib')
        cmake_prefix_path = build_options.get('CMAKE_PREFIX_PATH', None)
        if cmake_prefix_path:
            build_options["CMAKE_PREFIX_PATH"] = (
                cast(str, py_lib_path) + ";" + cast(str, cmake_prefix_path)
            )
        else:
            build_options['CMAKE_PREFIX_PATH'] = py_lib_path

        # Some options must be post-processed. Ideally, this list will be shrunk to only one or two options in the
        # future, as CMake can detect many of these libraries pretty comfortably. We have them here for now before CMake
        # integration is completed. They appear here not in the CMake.defines call below because they start with either
        # "BUILD_" or "USE_" and must be overwritten here.
        build_options.update({
            # Note: Do not add new build options to this dict if it is directly read from environment variable -- you
            # only need to add one in `CMakeLists.txt`. All build options that start with "BUILD_", "USE_", or "CMAKE_"
            # are automatically passed to CMake; For other options you can add to additional_options above.
            'BUILD_PYTHON': build_python,
            'BUILD_TEST': build_test,
        })

        # Options starting with CMAKE_
        cmake__options = {
            'CMAKE_INSTALL_PREFIX': install_dir,
        }

        # We set some CMAKE_* options in our Python build code instead of relying on the user's direct settings. Emit an
        # error if the user also attempts to set these CMAKE options directly.
        specified_cmake__options = set(build_options).intersection(cmake__options)
        if len(specified_cmake__options) > 0:
            print(', '.join(specified_cmake__options) +
                  ' should not be specified in the environment variable. They are directly set by PyTorch build script.')
        build_options.update(cmake__options)
        print(f'build_options: {build_options}')

        print(f'cmake_options: {cmake__options}')
        CMake.defines(args,
                      PYTHON_EXECUTABLE=sys.executable,
                      PYTHON_LIBRARY=cmake_python_library,
                      PYTHON_INCLUDE_DIR=sysconfig.get_path('include'),
                      TORCH_BUILD_VERSION=version,
                      **build_options)

        for env_var_name in my_env:
            if env_var_name.startswith('gh'):
                # github env vars use utf-8, on windows, non-ascii code may
                # cause problem, so encode first
                try:
                    my_env[env_var_name] = str(my_env[env_var_name].encode("utf-8"))
                except UnicodeDecodeError as e:
                    shex = ':'.join('{:02x}'.format(ord(c)) for c in my_env[env_var_name])
                    print('Invalid ENV[{}] = {}'.format(env_var_name, shex), file=sys.stderr)
                    print(e, file=sys.stderr)
        # According to the CMake manual, we should pass the arguments first,
        # and put the directory as the last element. Otherwise, these flags
        # may not be passed correctly.
        # Reference:
        # 1. https://cmake.org/cmake/help/latest/manual/cmake.1.html#synopsis
        # 2. https://stackoverflow.com/a/27169347
        args.append(base_dir)
        print(f'args: {args}')
        self.run(args, env=my_env)


    def build(self, my_env: Dict[str, str]) -> None:
        from .env import build_type

        build_args = ['--build', '.', '--target', 'install', '--config', build_type.build_type_string]

        # Determine the parallelism according to the following
        # priorities:
        # 1) MAX_JOBS environment variable
        # 2) If using the Ninja build system, delegate decision to it.
        # 3) Otherwise, fall back to the number of processors.

        # Allow the user to set parallelism explicitly. If unset,
        # we'll try to figure it out.
        max_jobs = os.getenv('MAX_JOBS')

        if max_jobs is not None:
            # Ninja is capable of figuring out the parallelism on its
            # own: only specify it explicitly if we are not using
            # Ninja.

            # This lists the number of processors available on the
            # machine. This may be an overestimate of the usable
            # processors if CPU scheduling affinity limits it
            # further. In the future, we should check for that with
            # os.sched_getaffinity(0) on platforms that support it.
            max_jobs = max_jobs or str(multiprocessing.cpu_count())

            # 3.12 becomes minimum, which provides a '-j' option:
            # build_args += ['-j', max_jobs] would be sufficient by
            # then. Until then, we use "--" to pass parameters to the
            # underlying build system.
            build_args += ['--']

            build_args += ['-j', max_jobs]
        self.run(build_args, my_env)


