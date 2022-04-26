
import os
import platform
import sys
import struct
from typing import Optional, cast

IS_LINUX = (platform.system() == 'Linux')

IS_64BIT = (struct.calcsize("P") == 8)

BUILD_DIR = 'build'

def check_env_flag(name: str, default: str = '') -> bool:
    return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']


def check_negative_env_flag(name: str, default: str = '') -> bool:
    return os.getenv(name, default).upper() in ['OFF', '0', 'NO', 'FALSE', 'N']


class BuildType(object):
    def __init__(self, cmake_build_type_env: Optional[str] = None) -> None:
        if cmake_build_type_env is not None:
            self.build_type_string = cmake_build_type_env
            return

        cmake_cache_txt = os.path.join(BUILD_DIR, 'CMakeCache.txt')
        if os.path.isfile(cmake_cache_txt):
            # Found CMakeCache.txt. Use the build type specified in it.
            from .cmake import get_cmake_cache_variables_from_file
            with open(cmake_cache_txt) as f:
                cmake_cache_vars = get_cmake_cache_variables_from_file(f)
            # Normally it is anti-pattern to determine build type from CMAKE_BUILD_TYPE because it is not used for
            # multi-configuration build tools, such as Visual Studio and XCode. But since we always communicate with
            # CMake using CMAKE_BUILD_TYPE from our Python scripts, this is OK here.
            self.build_type_string = cast(str, cmake_cache_vars['CMAKE_BUILD_TYPE'])
        else:
            self.build_type_string = os.environ.get('CMAKE_BUILD_TYPE', 'Release')

    def is_debug(self) -> bool:
        "Checks Debug build."
        return self.build_type_string == 'Debug'

    def is_rel_with_deb_info(self) -> bool:
        "Checks RelWithDebInfo build."
        return self.build_type_string == 'RelWithDebInfo'

    def is_release(self) -> bool:
        "Checks Release build."
        return self.build_type_string == 'Release'


# hotpatch environment variable 'CMAKE_BUILD_TYPE'. 'CMAKE_BUILD_TYPE' always prevails over DEBUG or REL_WITH_DEB_INFO.
if 'CMAKE_BUILD_TYPE' not in os.environ:
    if check_env_flag('DEBUG'):
        os.environ['CMAKE_BUILD_TYPE'] = 'Debug'
    elif check_env_flag('REL_WITH_DEB_INFO'):
        os.environ['CMAKE_BUILD_TYPE'] = 'RelWithDebInfo'
    else:
        os.environ['CMAKE_BUILD_TYPE'] = 'Release'

build_type = BuildType()