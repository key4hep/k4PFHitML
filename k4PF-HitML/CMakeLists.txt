#[[
Copyright (c) 2020-2024 Key4hep-Project.

This file is part of Key4hep.
See https://key4hep.github.io/key4hep-doc/ for further info.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
]]

file(GLOB _plugin_sources src/components/*.cpp)
gaudi_add_module(k4ProjectTemplatePlugins
                 SOURCES ${_plugin_sources}
                 LINK Gaudi::GaudiKernel
                      k4FWCore::k4FWCore
                      EDM4HEP::edm4hep
                      onnxruntime::onnxruntime
                      torch
                      )

install(TARGETS k4ProjectTemplatePlugins
  EXPORT ${CMAKE_PROJECT_NAME}Targets
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT bin
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT shlib
  PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/@{CMAKE_PROJECT_NAME}"
  COMPONENT dev)

include(CTest)

SET(test_name "test_PerformHitPF")
ExternalData_Add_Test(${test_name}
    NAME ${test_name}
    COMMAND sh +x k4PF-HitML/test/test_PerformHitPF.sh
            DATA{${CMAKE_SOURCE_DIR}/k4PF-HitML/test/models/clustering_test.onnx}
            DATA{${CMAKE_SOURCE_DIR}/k4PF-HitML/test/models/regression_neutral_test.onnx}
            DATA{${CMAKE_SOURCE_DIR}/k4PF-HitML/test/models/regression_charged_test.onnx})
set_tests_properties(${test_name} PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
ExternalData_Add_Target(${test_name})

get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

set(test_environment "\
LD_LIBRARY_PATH=\
${PROJECT_BINARY_DIR}:\
${PROJECT_BINARY_DIR}/${PROJECT_NAME}:\
${CMAKE_CURRENT_BINARY_DIR}:\
${CMAKE_CURRENT_BINARY_DIR}/genConfDir/${CURRENT_DIR_NAME}:\
$ENV{LD_LIBRARY_PATH};\
\
GAUDI_PLUGIN_PATH=\
${PROJECT_BINARY_DIR}:\
${PROJECT_BINARY_DIR}/${PROJECT_NAME}:\
${CMAKE_CURRENT_BINARY_DIR}:\
${CMAKE_CURRENT_BINARY_DIR}/genConfDir/${CURRENT_DIR_NAME}:\
$ENV{GAUDI_PLUGIN_PATH}:\
$ENV{LD_LIBRARY_PATH};\
\
PYTHONPATH=\
${PROJECT_BINARY_DIR}/${PROJECT_NAME}/genConfDir:\
${CMAKE_CURRENT_BINARY_DIR}/genConfDir:\
$ENV{PYTHONPATH};\
"
)
get_property(test_names DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY TESTS)
set_tests_properties(${test_names} PROPERTIES ENVIRONMENT "${test_environment}")

