if(ME_EB)
    target_sources(WarpX
      PRIVATE
        ElectrostaticSolver_EB.cpp
        WarpXPushFieldsEM.cpp
        WarpX_QED_Field_Pushers.cpp
    )
else()
    target_sources(WarpX
      PRIVATE
        ElectrostaticSolver.cpp
        WarpXPushFieldsEM.cpp
        WarpX_QED_Field_Pushers.cpp
    )
endif()



add_subdirectory(FiniteDifferenceSolver)
if(WarpX_PSATD)
    add_subdirectory(SpectralSolver)
endif()
