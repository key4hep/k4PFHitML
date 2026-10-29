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
#include "Shower.h"
#include "Helpers.h"

ShowerSplit splitShowersByTrackContent(const std::vector<Shower>& showers) {
  ShowerSplit out;
  for (size_t i = 0; i < showers.size(); ++i) {
    if (showers[i].getTracks().empty()) {
      out.neutral.push_back(i);
    } else {
      out.charged.push_back(i);
    }
  }
  return out;
}

std::pair<float, float> Shower::getCaloEnergy(const std::vector<edm4hep::CalorimeterHit> collection) const {

  float sum_e = 0;
  float sum_e_sq = 0;
  for (const auto& hit_i : collection) {
    sum_e += hit_i.getEnergy();
    sum_e_sq += std::pow(hit_i.getEnergy(), 2);
  }

  return {sum_e, sum_e_sq};
}

void Shower::addCalorimeterHit(const edm4hep::CalorimeterHit& hit, const std::string collection) {

  caloHits_.push_back(hit);

  if (collection == "ecal") {
    ecalHits_.push_back(hit);
    types_.push_back(2);
  } else if (collection == "hcal") {
    hcalHits_.push_back(hit);
    types_.push_back(3);
  } else if (collection == "muon") {
    muonHits_.push_back(hit);
    types_.push_back(4);
  }
}

float Shower::getTrackMomentum_mean(float bFieldTesla) {

  const float n_tracks = tracks_.size();

  if (n_tracks == 0) {
    return 0;
  }

  float p_sum = 0;
  for (const auto& track : tracks_) {
    p_sum +=
        momentumMagnitude(momentumFromTrackState(track.getTrackState(edm4hep::TrackState::AtIP).value(), bFieldTesla));
  }

  return p_sum / n_tracks;
}

float Shower::Chi2_mean() {

  const float n_tracks = tracks_.size();

  if (n_tracks == 0) {
    return 0;
  }

  float chi2_sum = 0;
  for (const auto& track : tracks_) {
    const float ndf = track.getNdf();
    chi2_sum += (ndf > 0.f) ? track.getChi2() / ndf : 0.f;
  }

  return chi2_sum / n_tracks;
}

const std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> Shower::get_pos() {

  std::vector<float> calo_x;
  std::vector<float> calo_y;
  std::vector<float> calo_z;

  for (const auto& hit_i : caloHits_) {
    auto pos = hit_i.getPosition();
    calo_x.push_back(pos.x);
    calo_y.push_back(pos.y);
    calo_z.push_back(pos.z);
  }

  std::vector<float> track_x;
  std::vector<float> track_y;
  std::vector<float> track_z;

  for (const auto& track_i : tracks_) {
    auto trackstate_calo = track_i.getTrackState(edm4hep::TrackState::AtCalorimeter).value();
    auto referencePoint_calo = trackstate_calo.referencePoint;
    track_x.push_back(referencePoint_calo.x);
    track_y.push_back(referencePoint_calo.y);
    track_z.push_back(referencePoint_calo.z);
  }

  std::vector<float> all_x;
  all_x.reserve(calo_x.size() + track_x.size());
  all_x.insert(all_x.end(), calo_x.begin(), calo_x.end());
  all_x.insert(all_x.end(), track_x.begin(), track_x.end());

  std::vector<float> all_y;
  all_y.reserve(calo_y.size() + track_y.size());
  all_y.insert(all_y.end(), calo_y.begin(), calo_y.end());
  all_y.insert(all_y.end(), track_y.begin(), track_y.end());

  std::vector<float> all_z;
  all_z.reserve(calo_z.size() + track_z.size());
  all_z.insert(all_z.end(), calo_z.begin(), calo_z.end());
  all_z.insert(all_z.end(), track_z.begin(), track_z.end());

  constexpr float norm = 3300.f;
  for (size_t i = 0; i < all_x.size(); ++i) {
    all_x[i] /= norm;
    all_y[i] /= norm;
    all_z[i] /= norm;
  }

  return {all_x, all_y, all_z};
}

const std::tuple<std::vector<float>, std::vector<float>> Shower::get_ep(float bFieldTesla) {

  std::vector<float> calo_e;
  std::vector<float> calo_p;

  for (const auto& hit_i : caloHits_) {
    calo_e.push_back(hit_i.getEnergy());
    calo_p.push_back(0);
  }

  std::vector<float> track_e;
  std::vector<float> track_p;

  for (const auto& track_i : tracks_) {
    const float p = momentumMagnitude(
        momentumFromTrackState(track_i.getTrackState(edm4hep::TrackState::AtIP).value(), bFieldTesla));

    track_e.push_back(0);
    track_p.push_back(p);
  }

  std::vector<float> all_e;
  all_e.reserve(calo_e.size() + track_e.size());
  all_e.insert(all_e.end(), calo_e.begin(), calo_e.end());
  all_e.insert(all_e.end(), track_e.begin(), track_e.end());

  std::vector<float> all_p;
  all_p.reserve(calo_p.size() + track_p.size());
  all_p.insert(all_p.end(), calo_p.begin(), calo_p.end());
  all_p.insert(all_p.end(), track_p.begin(), track_p.end());

  return {all_e, all_p};
}
