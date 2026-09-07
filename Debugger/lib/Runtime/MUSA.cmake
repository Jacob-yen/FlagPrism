include_guard(GLOBAL)

set(FLAGTREE_DEBUGGER_MUSA_ROOT ""
    CACHE PATH "Root directory of the MUSA toolkit used by the debugger runtime")

function(_flagtree_debugger_resolve_musa)
  set(_candidates)
  if(FLAGTREE_DEBUGGER_MUSA_ROOT)
    list(APPEND _candidates "${FLAGTREE_DEBUGGER_MUSA_ROOT}")
  endif()
  foreach(_env_name MUSA_HOME MUSA_PATH)
    if(DEFINED ENV{${_env_name}} AND NOT "$ENV{${_env_name}}" STREQUAL "")
      list(APPEND _candidates "$ENV{${_env_name}}")
    endif()
  endforeach()
  list(APPEND _candidates "/usr/local/musa")
  list(REMOVE_DUPLICATES _candidates)

  set(_resolved_include "")
  set(_resolved_library "")
  foreach(_candidate IN LISTS _candidates)
    if(NOT _candidate)
      continue()
    endif()
    if(EXISTS "${_candidate}/include/musa.h")
      set(_resolved_include "${_candidate}/include")
    endif()
    foreach(_library_dir "${_candidate}/lib" "${_candidate}/lib64")
      if(EXISTS "${_library_dir}/libmusa.so")
        set(_resolved_library "${_library_dir}/libmusa.so")
        break()
      endif()
    endforeach()
    if(_resolved_include AND _resolved_library)
      break()
    endif()
  endforeach()

  if(NOT _resolved_library)
    find_library(_system_musa NAMES musa PATHS
      /usr/lib/x86_64-linux-gnu /usr/lib64 /lib/x86_64-linux-gnu
      NO_DEFAULT_PATH)
    if(_system_musa)
      set(_resolved_library "${_system_musa}")
    endif()
  endif()

  if(_resolved_include AND _resolved_library)
    get_filename_component(_resolved_libdir "${_resolved_library}" DIRECTORY)
    set(FLAGTREE_DEBUGGER_MUSA_FOUND TRUE CACHE INTERNAL "" FORCE)
    set(FLAGTREE_DEBUGGER_MUSA_INCLUDE_DIR "${_resolved_include}" CACHE INTERNAL "" FORCE)
    set(FLAGTREE_DEBUGGER_MUSA_LIBRARY "${_resolved_library}" CACHE INTERNAL "" FORCE)
    set(FLAGTREE_DEBUGGER_MUSA_LIBRARY_DIR "${_resolved_libdir}" CACHE INTERNAL "" FORCE)
  else()
    set(FLAGTREE_DEBUGGER_MUSA_FOUND FALSE CACHE INTERNAL "" FORCE)
    set(FLAGTREE_DEBUGGER_MUSA_INCLUDE_DIR "" CACHE INTERNAL "" FORCE)
    set(FLAGTREE_DEBUGGER_MUSA_LIBRARY "" CACHE INTERNAL "" FORCE)
    set(FLAGTREE_DEBUGGER_MUSA_LIBRARY_DIR "" CACHE INTERNAL "" FORCE)
  endif()
endfunction()

function(flagtree_debugger_enable_musa target)
  _flagtree_debugger_resolve_musa()
  if(FLAGTREE_DEBUGGER_MUSA_FOUND)
    message(STATUS
      "FlagPrism Debugger: enabling MUSA runtime for ${target}")
    target_compile_definitions(${target}
      PRIVATE FLAGTREE_DEBUGGER_HAS_MUSA_RUNTIME=1)
    target_include_directories(${target}
      PRIVATE "${FLAGTREE_DEBUGGER_MUSA_INCLUDE_DIR}")
    target_link_libraries(${target}
      PUBLIC "${FLAGTREE_DEBUGGER_MUSA_LIBRARY}")
    target_link_directories(${target}
      PUBLIC "${FLAGTREE_DEBUGGER_MUSA_LIBRARY_DIR}")
    target_link_options(${target}
      PUBLIC "-Wl,-rpath-link,${FLAGTREE_DEBUGGER_MUSA_LIBRARY_DIR}")
    get_target_property(_target_type ${target} TYPE)
    if(NOT _target_type STREQUAL "OBJECT_LIBRARY")
      set_property(TARGET ${target} APPEND PROPERTY BUILD_RPATH
        "${FLAGTREE_DEBUGGER_MUSA_LIBRARY_DIR}")
    endif()
  else()
    message(STATUS
      "FlagPrism Debugger: MUSA runtime not found for ${target}, "
      "building stub MUSA adapter")
    target_compile_definitions(${target}
      PRIVATE FLAGTREE_DEBUGGER_HAS_MUSA_RUNTIME=0)
  endif()
endfunction()
