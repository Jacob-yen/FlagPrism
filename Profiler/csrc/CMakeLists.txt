add_profiler_library(FlagTreeProfilerBindings
  Profiler.cpp
)

if(FLAGPRISM_BACKEND STREQUAL "mthreads")
  add_profiler_library(FlagTreeProfilerMthreads
    lib/Driver/Mthreads/MthreadsApi.cpp
    lib/Profiler/Vendor/Adapter.cpp
    lib/Profiler/Vendor/MthreadsAdapter.cpp
    lib/Profiler/Vendor/MthreadsProfiler.cpp
    lib/Profiler/Vendor/Mode.cpp
  )
elseif(FLAGPRISM_BACKEND STREQUAL "ascend")
  add_profiler_library(FlagTreeProfilerCann
    lib/Driver/Ascend/AscendApi.cpp
    lib/Profiler/Vendor/Adapter.cpp
    lib/Profiler/Vendor/CannAdapter.cpp
    lib/Profiler/Vendor/CannProfiler.cpp
    lib/Profiler/Vendor/Mode.cpp
  )
else()
  add_profiler_library(FlagTreeProfilerCann
    lib/Driver/Ascend/AscendApi.cpp
    lib/Profiler/Vendor/Adapter.cpp
    lib/Profiler/Vendor/CannAdapter.cpp
    lib/Profiler/Vendor/CannProfiler.cpp
    lib/Profiler/Vendor/Mode.cpp
  )
  add_profiler_library(FlagTreeProfilerMthreads
    lib/Driver/Mthreads/MthreadsApi.cpp
    lib/Profiler/Vendor/Adapter.cpp
    lib/Profiler/Vendor/MthreadsAdapter.cpp
    lib/Profiler/Vendor/MthreadsProfiler.cpp
    lib/Profiler/Vendor/Mode.cpp
  )
endif()

add_subdirectory(lib)
