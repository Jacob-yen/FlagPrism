include_guard(GLOBAL)

# TOPS memory-transfer dependencies shared by bundled and standalone builds.
function(flagtree_debugger_enable_enflame target)
  find_path(FLAGTREE_TOPS_INCLUDE_DIR tops/tops_runtime.h
    HINTS "$ENV{TOPS_HOME}/include" "$ENV{CAPS_PATH}/include" "/opt/tops/include")
  find_library(FLAGTREE_TOPS_RUNTIME topsrt
    HINTS "$ENV{TOPS_HOME}/lib" "$ENV{CAPS_PATH}/lib" "/opt/tops/lib")
  if(NOT FLAGTREE_TOPS_INCLUDE_DIR OR NOT FLAGTREE_TOPS_RUNTIME)
    message(FATAL_ERROR "Enflame runtime requires TOPS headers and libtopsrt")
  endif()
  target_compile_definitions(${target} PRIVATE FLAGTREE_DEBUGGER_HAS_ENFLAME_RUNTIME=1)
  target_include_directories(${target} PRIVATE "${FLAGTREE_TOPS_INCLUDE_DIR}")
  target_link_libraries(${target} PRIVATE "${FLAGTREE_TOPS_RUNTIME}")
endfunction()
