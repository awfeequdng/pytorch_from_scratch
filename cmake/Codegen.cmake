
################################################################################
# Helper functions
################################################################################

function(filter_list output input)
    unset(result)
    foreach(filename ${${input}})
        foreach(pattern ${ARGN})
            if("${filename}" MATCHES "${pattern}")
                list(APPEND result "${filename}")
            endif()
        endforeach()
    endforeach()
    set(${output} ${result} PARENT_SCOPE)
endfunction()

function(filter_list_exclude output input)
    unset(result)
    foreach(filename ${${input}})
        foreach(pattern ${ARGN})
            if(NOT "${filename}" MATCHES "${pattern}")
                list(APPEND result "${filename}")
            endif()
        endforeach()
    endforeach()
    set(${output} ${result} PARENT_SCOPE)
endfunction()

# ---[ Write the macros file
# configure_file(
#     ${CMAKE_CURRENT_LIST_DIR}/../caffe2/core/macros.h.in
#     ${CMAKE_BINARY_DIR}/caffe2/core/macros.h)

# ---[ Installing the header files
# install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/../caffe2
#         DESTINATION include
#         FILES_MATCHING PATTERN "*.h")


function(append_filelist name outputvar)
  set(_rootdir "${${CMAKE_PROJECT_NAME}_SOURCE_DIR}/")
  # configure_file adds its input to the list of CMAKE_RERUN dependencies
  configure_file(
      ${PROJECT_SOURCE_DIR}/tools/build_variables.bzl
      ${PROJECT_BINARY_DIR}/caffe2/build_variables.bzl)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "exec(open('${PROJECT_SOURCE_DIR}/tools/build_variables.bzl').read());print(';'.join(['${_rootdir}' + x for x in ${name}]))"
    WORKING_DIRECTORY "${_rootdir}"
    RESULT_VARIABLE _retval
    OUTPUT_VARIABLE _tempvar)

  message("PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")
  message("_rootdir: ${_rootdir}")
  message("_tempvar: ${_tempvar}")
  if(NOT _retval EQUAL 0)
    message(FATAL_ERROR "Failed to fetch filelist ${name} from build_variables.bzl")
  endif()
  string(REPLACE "\n" "" _tempvar "${_tempvar}")
  list(APPEND ${outputvar} ${_tempvar})
  set(${outputvar} "${${outputvar}}" PARENT_SCOPE)
endfunction()