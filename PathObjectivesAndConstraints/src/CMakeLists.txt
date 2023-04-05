add_library(${PROJECT_NAME} SHARED 
    BsplineToMinvo.cpp
    BsplineToMinvo.hpp
    CubicEquationSolver.cpp
    CubicEquationSolver.hpp
    CBindingHelper.cpp
    CBindingHelper.hpp
    MDMAlgorithmClass.cpp
    MDMAlgorithmClass.hpp
    DerivativeEvaluator.cpp 
    DerivativeEvaluator.hpp
    ThirdOrderCurvatureBounds.cpp
    ThirdOrderCurvatureBounds.hpp
    WaypointConstraints.cpp
    WaypointConstraints.hpp
    ObjectiveFunctions.cpp
    ObjectiveFunctions.hpp
    ConvexHullCollisionChecker.cpp
    ConvexHullCollisionChecker.hpp
    ObstacleConstraints.cpp
    ObstacleConstraints.hpp
    CurvatureConstraints.cpp
    CurvatureConstraints.hpp)

target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")