/*
 * Copyright (c) 2020-2024 Key4hep-Project.
 *
 * This file is part of Key4hep.
 * See https://key4hep.github.io/key4hep-doc/ for further info.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef PFPARTICLEBUILDER_H
#define PFPARTICLEBUILDER_H

#include <algorithm>
#include <cmath>
#include <limits>

#include "ONNXHelper.h"
#include "edm4hep/ClusterCollection.h"
#include "edm4hep/ParticleIDCollection.h"
#include "edm4hep/ReconstructedParticleCollection.h"
#include "edm4hep/Track.h"
#include "edm4hep/Vector3f.h"
#include <vector>

class Shower;

struct ParticleRecoInfo {
  edm4hep::Vector3f momentum{};
  edm4hep::Vector3f referencePoint{};
  edm4hep::Vector3f direction{};

  float energy{0.f};
  float mass{0.f};
  float pidScore{0.f};
  int physicsClass{0};
  float charge{0.f};
  int pdg{0};

  std::vector<edm4hep::Track> tracks{};

  std::vector<float> pidScores{};
};

struct PIDPrediction {
  int localIndex{0};           // index inside the NN output, e.g. 0,1,2
  int physicsClass{0};         // mapped class, e.g. 0,1,4 or 2,3
  float score{0.f};            // softmax probability of winning class
  std::vector<float> scores{}; // full softmax vector, one entry per class
};

const std::vector<float>& findPIDOutput(const ONNXHelper& model, const std::vector<std::vector<float>>& outputs);

PIDPrediction decodePIDLogits(const std::vector<float>& logits, const std::vector<int>& classMap);

ParticleRecoInfo buildChargedRecoInfo(const Shower& shower, int predictedClass, float pidScore,
                                      float bFieldTesla = 2.0f, bool reassignLowPMuons = true,
                                      float muonToChargedHadronPThreshold = 1.0f);

ParticleRecoInfo buildNeutralRecoInfo(const Shower& shower, int predictedClass, float pidScore, float predictedEnergy,
                                      const edm4hep::Vector3f& predictedDirection,
                                      const edm4hep::Vector3f& predictedReferencePoint);

edm4hep::Vector3f computeNeutralReferencePoint(const Shower& shower);

edm4hep::Vector3f computeNeutralPandoraReferencePoint(const Shower& shower);

edm4hep::Vector3f computeNeutralDirection(const edm4hep::Vector3f& referencePoint);

void fillRecoParticle(edm4hep::ReconstructedParticleCollection& particles, edm4hep::ParticleIDCollection& pids,
                      edm4hep::ClusterCollection& clusters, const Shower& shower, const ParticleRecoInfo& info);

#endif
