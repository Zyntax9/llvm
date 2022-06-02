list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(LLVMCheckLinkerFlag)

# add_stripped_pdb(TARGET_NAME)
#
# Will add option for generating stripped PDB file and install the generated
# file as ${ARG_TARGET_NAME}.pdb in bin folder.
# NOTE: LLD does not currently support /PDBSTRIPPED so the PDB file is optional.
macro(add_stripped_pdb ARG_TARGET_NAME)
  get_filename_component(FULL_STRIPPED_PDB_FILE_PATH
                         "${ARG_TARGET_NAME}.stripped.pdb" ABSOLUTE
                         BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  llvm_check_linker_flag(CXX "/PDBSTRIPPED:${FULL_STRIPPED_PDB_FILE_PATH}"
                         LINKER_SUPPORTS_PDBSTRIPPED)
  if(LINKER_SUPPORTS_PDBSTRIPPED)
    target_link_options(${ARG_TARGET_NAME}
                        PRIVATE "/PDBSTRIPPED:${FULL_STRIPPED_PDB_FILE_PATH}")
    install(FILES "${FULL_STRIPPED_PDB_FILE_PATH}"
            DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
            RENAME "${ARG_TARGET_NAME}.pdb"
            COMPONENT ${ARG_TARGET_NAME}
            OPTIONAL)
  endif()
endmacro()
