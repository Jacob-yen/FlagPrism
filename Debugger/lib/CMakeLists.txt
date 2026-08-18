include(${CMAKE_CURRENT_SOURCE_DIR}/Runtime/CANN.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/Runtime/MUSA.cmake)

add_triton_library(FlagTreeDebuggerRuntime
  Runtime/BufferLayout.cpp
  Runtime/TransferEngine.cpp
)
flagtree_debugger_enable_cann(FlagTreeDebuggerRuntime)
flagprism_enable_debugger_runtime(FlagTreeDebuggerRuntime)

add_triton_library(FlagTreeDebuggerFrontend
  Frontend/Bridge.cpp
  Metadata/TrackedOpTable.cpp

  LINK_LIBS PUBLIC
  FlagTreeDebuggerRuntime
  LLVMSupport
)
flagtree_debugger_enable_cann(FlagTreeDebuggerFrontend)
flagprism_enable_debugger_runtime(FlagTreeDebuggerFrontend)

add_triton_library(FlagTreeDebugger
  IR/Dialect.cpp
  IR/Ops.cpp
  Metadata/Passes.cpp
  Instrumentation/Collectors.cpp
  Instrumentation/Passes.cpp
  Instrumentation/RecordBuilder.cpp
  Instrumentation/Writer.cpp
  Decode/Decoder.cpp
  Decode/Reporter.cpp

  DEPENDS
  FlagTreeDebugTableGen

  LINK_LIBS PUBLIC
  FlagTreeDebuggerFrontend
  FlagTreeDebuggerRuntime
  MLIRArithDialect
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRPtrDialect
  MLIRTensorDialect
  TritonIR
)
flagtree_debugger_enable_cann(FlagTreeDebugger)
flagprism_enable_debugger_runtime(FlagTreeDebugger)
