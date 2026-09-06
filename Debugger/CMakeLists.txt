# FlagPrism Debugger is optional so regular FlagTree builds can omit all of its
# native sources, generated IR, runtime support, and bindings.
add_subdirectory(include/Debugger)
add_subdirectory(lib)

if(TRITON_BUILD_UT)
  add_subdirectory(test/unittest)
endif()
