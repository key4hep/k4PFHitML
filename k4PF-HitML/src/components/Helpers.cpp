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
#include "Helpers.h"

#include "Gaudi/Property.h"
#include "edm4hep/ReconstructedParticleCollection.h"
#include <torch/torch.h>

float mean_var(std::vector<float> obs) {

  int n_hits = obs.size();
  float x_sum = 0;

  for (int i = 0; i < n_hits; ++i) {
    x_sum += obs[i];
  }

  return x_sum / n_hits;
}

// var = E(x^2) - E(x)^2
float disperion(Shower shower_i, std::vector<edm4hep::CalorimeterHit> collection) {

  float sys_e = shower_i.getCaloEnergy(collection).first;
  float sys_e_sq = shower_i.getCaloEnergy(collection).second;

  float n_sys_hits = collection.size();

  float dispersion = sys_e_sq / n_sys_hits - std::pow((sys_e / n_sys_hits), 2);

  return dispersion;
}

edm4hep::Vector3f momentumFromTrackState(const edm4hep::TrackState& ts, float bFieldTesla) {
  // speed of light in units of GeV / (T mm)
  constexpr float c_light = 2.99792e-4f;
  if (ts.omega == 0.f) {
    return {0.f, 0.f, 0.f};
  }
  const float pt = c_light * std::abs(bFieldTesla / ts.omega);
  return {std::cos(ts.phi) * pt, std::sin(ts.phi) * pt, ts.tanLambda * pt};
}

std::vector<std::vector<float>> one_hot_encode(std::vector<int> values, int n_classes) {
  std::vector<std::vector<float>> out(values.size(), std::vector<float>(n_classes, 0.0f));
  for (std::size_t i = 0; i < values.size(); ++i) {
    int idx = values[i];
    if (idx >= 0 && idx < n_classes)
      out[i][idx] = 1.0f;
  }
  return out;
}
