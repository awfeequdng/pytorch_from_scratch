
################################################################################################
# Clears variables from list
# Usage:
#   caffe_clear_vars(<variables_list>)
macro(caffe_clear_vars)
  foreach(_var ${ARGN})
    unset(${_var})
  endforeach()
endmacro()

################################################################################################
# Prints list element per line
# Usage:
#   caffe_print_list(<list>)
function(caffe_print_list)
  foreach(e ${ARGN})
    message(STATUS ${e})
  endforeach()
endfunction()

################################################################################################
# Parses a version string that might have values beyond major, minor, and patch
# and set version variables for the library.
# Usage:
#   caffe2_parse_version_str(<library_name> <version_string>)
function(caffe2_parse_version_str LIBNAME VERSIONSTR)
  string(REGEX REPLACE "^([0-9]+).*$" "\\1" ${LIBNAME}_VERSION_MAJOR "${VERSIONSTR}")
  string(REGEX REPLACE "^[0-9]+\\.([0-9]+).*$" "\\1" ${LIBNAME}_VERSION_MINOR  "${VERSIONSTR}")
  string(REGEX REPLACE "[0-9]+\\.[0-9]+\\.([0-9]+).*$" "\\1" ${LIBNAME}_VERSION_PATCH "${VERSIONSTR}")
  set(${LIBNAME}_VERSION_MAJOR ${${LIBNAME}_VERSION_MAJOR} ${ARGN} PARENT_SCOPE)
  set(${LIBNAME}_VERSION_MINOR ${${LIBNAME}_VERSION_MINOR} ${ARGN} PARENT_SCOPE)
  set(${LIBNAME}_VERSION_PATCH ${${LIBNAME}_VERSION_PATCH} ${ARGN} PARENT_SCOPE)
  set(${LIBNAME}_VERSION "${${LIBNAME}_VERSION_MAJOR}.${${LIBNAME}_VERSION_MINOR}.${${LIBNAME}_VERSION_PATCH}" PARENT_SCOPE)
endfunction()


##############################################################################
# Macro to update cached options.
macro(caffe2_update_option variable value)
  if(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
    get_property(__help_string CACHE ${variable} PROPERTY HELPSTRING)
    set(${variable} ${value} CACHE BOOL ${__help_string} FORCE)
  else()
    set(${variable} ${value})
  endif()
endmacro()


##############################################################################
# Add an interface library definition that is dependent on the source.
#
# It's probably easiest to explain why this macro exists, by describing
# what things would look like if we didn't have this macro.
#
# Let's suppose we want to statically link against torch.  We've defined
# a library in cmake called torch, and we might think that we just
# target_link_libraries(my-app PUBLIC torch).  This will result in a
# linker argument 'libtorch.a' getting passed to the linker.
#
# Unfortunately, this link command is wrong!  We have static
# initializers in libtorch.a that would get improperly pruned by
# the default link settings.  What we actually need is for you
# to do -Wl,--whole-archive,libtorch.a -Wl,--no-whole-archive to ensure
# that we keep all symbols, even if they are (seemingly) not used.
#
# What caffe2_interface_library does is create an interface library
# that indirectly depends on the real library, but sets up the link
# arguments so that you get all of the extra link settings you need.
# The result is not a "real" library, and so we have to manually
# copy over necessary properties from the original target.
#
# (The discussion above is about static libraries, but a similar
# situation occurs for dynamic libraries: if no symbols are used from
# a dynamic library, it will be pruned unless you are --no-as-needed)
macro(caffe2_interface_library SRC DST)
  add_library(${DST} INTERFACE)
  add_dependencies(${DST} ${SRC})
  # Depending on the nature of the source library as well as the compiler,
  # determine the needed compilation flags.
  get_target_property(__src_target_type ${SRC} TYPE)
  # Depending on the type of the source library, we will set up the
  # link command for the specific SRC library.
  if(${__src_target_type} STREQUAL "STATIC_LIBRARY")
    # In the case of static library, we will need to add whole-static flags.
    if(APPLE)
      target_link_libraries(
          ${DST} INTERFACE -Wl,-force_load,\"$<TARGET_FILE:${SRC}>\")
    else()
      # Assume everything else is like gcc
      target_link_libraries(${DST} INTERFACE
          "-Wl,--whole-archive,\"$<TARGET_FILE:${SRC}>\" -Wl,--no-whole-archive")
    endif()
    # Link all interface link libraries of the src target as well.
    # For static library, we need to explicitly depend on all the libraries
    # that are the dependent library of the source library. Note that we cannot
    # use the populated INTERFACE_LINK_LIBRARIES property, because if one of the
    # dependent library is not a target, cmake creates a $<LINK_ONLY:src> wrapper
    # and then one is not able to find target "src". For more discussions, check
    #   https://gitlab.kitware.com/cmake/cmake/issues/15415
    #   https://cmake.org/pipermail/cmake-developers/2013-May/019019.html
    # Specifically the following quote
    #
    # """
    # For STATIC libraries we can define that the PUBLIC/PRIVATE/INTERFACE keys
    # are ignored for linking and that it always populates both LINK_LIBRARIES
    # LINK_INTERFACE_LIBRARIES.  Note that for STATIC libraries the
    # LINK_LIBRARIES property will not be used for anything except build-order
    # dependencies.
    # """
    target_link_libraries(${DST} INTERFACE
        $<TARGET_PROPERTY:${SRC},LINK_LIBRARIES>)
  elseif(${__src_target_type} STREQUAL "SHARED_LIBRARY")
    if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
      if (APPLE)
        target_link_libraries(${DST} INTERFACE ${SRC})
      else()
        target_link_libraries(${DST} INTERFACE
            "-Wl,--no-as-needed,\"$<TARGET_FILE:${SRC}>\" -Wl,--as-needed")
      endif()
    else()
      target_link_libraries(${DST} INTERFACE ${SRC})
    endif()
    # Link all interface link libraries of the src target as well.
    # For shared libraries, we can simply depend on the INTERFACE_LINK_LIBRARIES
    # property of the target.
    target_link_libraries(${DST} INTERFACE
        $<TARGET_PROPERTY:${SRC},INTERFACE_LINK_LIBRARIES>)
  else()
    message(FATAL_ERROR
        "You made a CMake build file error: target " ${SRC}
        " must be of type either STATIC_LIBRARY or SHARED_LIBRARY. However, "
        "I got " ${__src_target_type} ".")
  endif()
  # For all other interface properties, manually inherit from the source target.
  set_target_properties(${DST} PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS
    $<TARGET_PROPERTY:${SRC},INTERFACE_COMPILE_DEFINITIONS>
    INTERFACE_COMPILE_OPTIONS
    $<TARGET_PROPERTY:${SRC},INTERFACE_COMPILE_OPTIONS>
    INTERFACE_INCLUDE_DIRECTORIES
    $<TARGET_PROPERTY:${SRC},INTERFACE_INCLUDE_DIRECTORIES>
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
    $<TARGET_PROPERTY:${SRC},INTERFACE_SYSTEM_INCLUDE_DIRECTORIES>)
endmacro()

##############################################################################
# Add standard compile options.
# Usage:
#   torch_compile_options(lib_name)
function(torch_compile_options libname)
  set_property(TARGET ${libname} PROPERTY CXX_STANDARD 20)
  set(private_compile_options "")

  # ---[ Check if warnings should be errors.
  if(WERROR)
    list(APPEND private_compile_options -Werror)
  endif()

  # until they can be unified, keep these lists synced with setup.py
  list(APPEND private_compile_options
        -Wall
        -Wextra
        -Wno-unused-parameter
        -Wno-unused-function
        -Wno-unused-result
        -Wno-unused-local-typedefs
        -Wno-missing-field-initializers
        -Wno-write-strings
        -Wno-unknown-pragmas
        -Wno-type-limits
        -Wno-array-bounds
        -Wno-unknown-pragmas
        -Wno-sign-compare
        -Wno-strict-overflow
        -Wno-strict-aliasing
        -Wno-error=deprecated-declarations
        # Clang has an unfixed bug leading to spurious missing braces
        # warnings, see https://bugs.llvm.org/show_bug.cgi?id=21629
        -Wno-missing-braces
        )
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    list(APPEND private_compile_options
          -Wno-range-loop-analysis)
  else()
    list(APPEND private_compile_options
          # Considered to be flaky.  See the discussion at
          # https://github.com/pytorch/pytorch/pull/9608
      -Wno-maybe-uninitialized)
  endif()

  target_compile_options(${libname} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:${private_compile_options}>)
  if(USE_CUDA)
    string(FIND "${private_compile_options}" " " space_position)
    if(NOT space_position EQUAL -1)
      message(FATAL_ERROR "Found spaces in private_compile_options='${private_compile_options}'")
    endif()
    # Convert CMake list to comma-separated list
    string(REPLACE ";" "," private_compile_options "${private_compile_options}")
    target_compile_options(${libname} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${private_compile_options}>)
  endif()

  if(NOT USE_ASAN)
    # Enable hidden visibility by default to make it easier to debug issues with
    # TORCH_API annotations. Hidden visibility with selective default visibility
    # behaves close enough to Windows' dllimport/dllexport.
    #
    # Unfortunately, hidden visibility messes up some ubsan warnings because
    # templated classes crossing library boundary get duplicated (but identical)
    # definitions. It's easier to just disable it.
    target_compile_options(${libname} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>: -fvisibility=hidden>)
  endif()

  # Use -O2 for release builds (-O3 doesn't improve perf, and -Os results in perf regression)
  target_compile_options(${libname} PRIVATE
      $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-O2>)

endfunction()


##############################################################################
# Multiplex between adding libraries for CUDA
# Usage:
#   torch_cuda_based_add_library(cuda_target)
#
macro(torch_cuda_based_add_library cuda_target)
  if(USE_CUDA)
    add_library(${cuda_target} ${ARGN})
  endif()
endmacro()