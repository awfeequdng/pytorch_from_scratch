from typing import Optional, Dict

import shutil
import os

from .setup_helpers.env import check_negative_env_flag
from .setup_helpers.cmake import CMake


def _create_build_env() -> Dict[str, str]:
    # XXX - our cmake file sometimes looks at the system environment
    # and not cmake flags!
    # you should NEVER add something to this list. It is bad practice to
    # have cmake read the environment
    my_env = os.environ.copy()
    if 'CUDA_HOME' in my_env:  # Keep CUDA_HOME. This env variable is still used in other part.
        my_env['CUDA_BIN_PATH'] = my_env['CUDA_HOME']

    return my_env

def build_caffe2(
    version: Optional[str],
    cmake_python_library: Optional[str],
    build_python: bool,
    rerun_cmake: bool,
    cmake_only: bool,
    cmake: CMake,
) -> None:
    my_env = _create_build_env()
    print(f'my_env: {my_env}')
    build_test = not check_negative_env_flag('BUILD_TEST')
    cmake.generate(version,
                   cmake_python_library,
                   build_python,
                   build_test,
                   my_env,
                   rerun_cmake)
    if cmake_only:
        return
    print(f'build_caffe2: {my_env}, \ncmake.build_dir: {cmake.build_dir}')
    import sys
    cmake.build(my_env)

    # if build_python:
    #     caffe2_proto_dir = os.path.join(cmake.build_dir, 'caffe2', 'proto')
    #     for proto_file in glob(os.path.join(caffe2_proto_dir, '*.py')):
    #         if proto_file != os.path.join(caffe2_proto_dir, '__init__.py'):
    #             shutil.copy(proto_file, os.path.join('caffe2', 'proto'))