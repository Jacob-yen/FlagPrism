set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
set(FLAGTREE_DEBUGGER_INCLUDE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

set(LLVM_TARGET_DEFINITIONS FlagTreeDebugOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls
              -I "${FLAGTREE_DEBUGGER_INCLUDE_ROOT}")
mlir_tablegen(Ops.cpp.inc -gen-op-defs
              -I "${FLAGTREE_DEBUGGER_INCLUDE_ROOT}")
add_mlir_doc(FlagTreeDebugOps FlagTreeDebugOps dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS FlagTreeDebugDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=flagtree_debug
              -I "${FLAGTREE_DEBUGGER_INCLUDE_ROOT}")
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=flagtree_debug
              -I "${FLAGTREE_DEBUGGER_INCLUDE_ROOT}")

add_public_tablegen_target(FlagTreeDebugTableGen)
