add_library(${PROJECT_NAME} SHARED 
    BsplineToMinvo.cpp
    CubicEquationSolver.cpp
    CBindHelperFunctions.cpp
    RotationHelperFunctions.cpp
    MDMAlgorithmClass.cpp
    DerivativeEvaluator.cpp 
    ObstacleConstraints.cpp
    DerivativeBounds.cpp
    ControlPointDerivativeBounds.cpp
    CrossTermEvaluator.cpp
    CrossTermProperties.cpp
    CrossTermBounds.cpp
    SphereCollisionEvaluator.cpp)
    # BsplineToMinvo.hpp
    # CubicEquationSolver.hpp
    # CBindHelperFunctions.hpp
    # RotationHelperFunctions.hpp
    # MDMAlgorithmClass.hpp
    # DerivativeEvaluator.hpp
    # ObstacleConstraints.hpp
    # DerivativeBounds.hpp
    # ControlPointDerivativeBounds.hpp
    # CrossTermEvaluator.hpp
    # CrossTermProperties.hpp
    # CrossTermBounds.hpp
    # SphereCollisionEvaluator.hpp)

target_include_directories(${PROJECT_NAME} PUBLIC 
"${CMAKE_CURRENT_SOURCE_DIR}/src"
 "../include")
