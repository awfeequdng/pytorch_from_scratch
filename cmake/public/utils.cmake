
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