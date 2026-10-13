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

#ifndef SHOWER_H
#define SHOWER_H

// edm4hep imports
#include "edm4hep/CalorimeterHitCollection.h"
#include "edm4hep/TrackCollection.h"

// shower object, assig energy, direction, PID..

class Shower {
public:
  void addCalorimeterHit(const edm4hep::CalorimeterHit& hit, const std::string collection = "");

  void addTrack(const edm4hep::Track& track) {
    tracks_.push_back(track);
    types_.push_back(1);
  }

  void addBetas(float beta) { betas_.push_back(beta); }

  const std::vector<edm4hep::CalorimeterHit>& getCalorimeterHits() const { return caloHits_; }

  const std::vector<edm4hep::Track>& getTracks() const { return tracks_; }

  std::pair<float, float> getCaloEnergy(const std::vector<edm4hep::CalorimeterHit> collection) const;

  float getTrackMomentum_mean(float bFieldTesla = 2.0f); // retruns mean track momentum per shower

  float Chi2_mean();

  const std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> get_pos();

  const std::tuple<std::vector<float>, std::vector<float>> get_ep(float bFieldTesla = 2.0f);

  std::vector<edm4hep::CalorimeterHit> caloHits_;
  std::vector<edm4hep::CalorimeterHit> ecalHits_;
  std::vector<edm4hep::CalorimeterHit> hcalHits_;
  std::vector<edm4hep::CalorimeterHit> muonHits_;
  std::vector<edm4hep::Track> tracks_;

  std::vector<int> types_;
  std::vector<float> betas_;
  int64_t label_ = -1; // DPC cluster label this shower was built from (validation/debugging)
};

struct ShowerSplit {
  std::vector<size_t> charged;
  std::vector<size_t> neutral;
};

ShowerSplit splitShowersByTrackContent(const std::vector<Shower>& showers);

#endif // SHOWER_H
