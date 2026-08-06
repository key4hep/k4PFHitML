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
CMAKE_MINIMUM_REQUIRED(VERSION 3.12)


project(k4ProjectTemplate LANGUAGES CXX)

# please keep this layout for version setting, used by the automatic tagging script
set(PACKAGE_VERSION_MAJOR 0)
set(PACKAGE_VERSION_MINOR 1)
set(PACKAGE_VERSION_PATCH 0)

set(PACKAGE_VERSION "${PACKAGE_VERSION_MAJOR}.${PACKAGE_VERSION_MINOR}.${PACKAGE_VERSION_PATCH}")

# dependencies are declared like this
find_package(ROOT COMPONENTS RIO Tree)
find_package(EDM4HEP)
find_package(k4FWCore)
find_package(Gaudi)

# make sure the onnx find script is found:
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(onnxruntime REQUIRED)
find_package(Torch REQUIRED)
if(onnxruntime_FOUND)
  message("ONNX Runtime inference library found.")
else()
  message("ONNX Runtime not found!")
endif()
#---------------------------------------------------------------

include(cmake/Key4hepConfig.cmake)

include(GNUInstallDirs)

include(CTest)

include(ExternalData)
list(APPEND ExternalData_URL_TEMPLATES
  "https://key4hep.web.cern.ch:443/testFiles/k4PFHitML/%(hash)"
)

add_subdirectory(k4PF-HitML)

include(cmake/CreateProjectConfig.cmake)
