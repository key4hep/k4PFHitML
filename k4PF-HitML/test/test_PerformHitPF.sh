#!/bin/bash
##
## Copyright (c) 2020-2024 Key4hep-Project.
##
## This file is part of Key4hep.
## See https://key4hep.github.io/key4hep-doc/ for further info.
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##
set -e


MODEL_CLUSTERING=$1
MODEL_REGRESSION_NEUTRAL=$2
MODEL_REGRESSION_CHARGED=$3

NEV=5
SIM_FILE=k4PF-HitML/test/out_sim_edm4hep_test.root
REC_BASENAME=k4PF-HitML/test/out_reco_edm4hep_test
REC_FILE=${REC_BASENAME}_REC.edm4hep.root
OUTPUT_FILE=k4PF-HitML/test/output_HitPF_test.edm4hep.root
rm -f ${SIM_FILE} ${REC_FILE} ${OUTPUT_FILE}


CLDGEO=CLD_o2_v07
XML_FILE=${K4GEO}/FCCee/CLD/compact/${CLDGEO}/${CLDGEO}.xml


CLDCONFIG_DIR=k4PF-HitML/test/CLDConfig
if [ ! -d "${CLDCONFIG_DIR}" ]; then
  git clone --depth 1 https://github.com/key4hep/CLDConfig.git "${CLDCONFIG_DIR}"
fi
STEERING_FILE=${CLDCONFIG_DIR}/CLDConfig/cld_steer.py

ddsim --steeringFile ${STEERING_FILE} \
      --compactFile ${XML_FILE} \
      -G --gun.distribution uniform --gun.particle pi- \
      --random.seed 42 \
      --numberOfEvents ${NEV} \
      --outputFile ${SIM_FILE}


SIM_FILE_ABS="$(pwd)/${SIM_FILE}"
REC_BASENAME_ABS="$(pwd)/${REC_BASENAME}"
(cd "${CLDCONFIG_DIR}/CLDConfig" && k4run CLDReconstruction.py -n ${NEV} \
  --inputFiles "${SIM_FILE_ABS}" \
  --outputBasename "${REC_BASENAME_ABS}")

k4run k4PF-HitML/options/PerformMLPF.py \
  --inputFiles "${REC_FILE}" \
  --compactFile "${XML_FILE}" \
  --onnx_model_clustering "${MODEL_CLUSTERING}" \
  --onnx_model_properties_neutral "${MODEL_REGRESSION_NEUTRAL}" \
  --onnx_model_properties_charged "${MODEL_REGRESSION_CHARGED}" \
  --outputFile ${OUTPUT_FILE} \
  --num_ev ${NEV}

python3 k4PF-HitML/test/check_output.py ${OUTPUT_FILE}
