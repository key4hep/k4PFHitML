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
#ifndef DATAPREPROCESSING_H
#define DATAPREPROCESSING_H

// edm4hep imports
#include "edm4hep/CalorimeterHitCollection.h"
#include "edm4hep/MCParticleCollection.h"
#include "edm4hep/ReconstructedParticleCollection.h"
#include "edm4hep/TrackCollection.h"
#include "edm4hep/TrackerHit.h"

#include "ONNXHelper.h"
#include <torch/torch.h>

#include "Shower.h"

struct PreprocessedData {
  std::map<std::string, std::vector<float>> features;
  std::vector<std::array<int64_t, 3>> hit_mapping;
};

struct ClusteringInputs {
  std::vector<ONNXInput> inputs;
  unsigned long long batch_size;
};

struct PropertyInputs {
  std::vector<ONNXInput> inputs;
};

class DataPreprocessing {
public:
  DataPreprocessing(const edm4hep::CalorimeterHitCollection& EcalBarrel_hits,
                    const edm4hep::CalorimeterHitCollection& HcalBarrel_hits,
                    const edm4hep::CalorimeterHitCollection& EcalEndcap_hits,
                    const edm4hep::CalorimeterHitCollection& HcalEndcap_hits,
                    const edm4hep::CalorimeterHitCollection& HcalOther_hits,
                    const edm4hep::CalorimeterHitCollection& Muon_hits, const edm4hep::TrackCollection& tracks,
                    float bFieldTesla = 2.0f);

  PreprocessedData extract() const;

  ClusteringInputs convertModelInputs(std::map<std::string, std::vector<float>> features) const;
  std::vector<PropertyInputs> prepare_prop(std::vector<Shower> shower) const;

  // helper function to return collections
  const edm4hep::CalorimeterHitCollection& ecalBarrel() const { return ecalbarrel_; }
  const edm4hep::CalorimeterHitCollection& ecalEndcap() const { return ecalendcap_; }
  const edm4hep::CalorimeterHitCollection& hcalBarrel() const { return hcalbarrel_; }
  const edm4hep::CalorimeterHitCollection& hcalEndcap() const { return hcalendcap_; }
  const edm4hep::CalorimeterHitCollection& hcalOther() const { return hcalother_; }
  const edm4hep::CalorimeterHitCollection& muons() const { return muons_; }
  const edm4hep::TrackCollection& tracks() const { return tracks_; }

private:
  const edm4hep::CalorimeterHitCollection& ecalbarrel_;
  const edm4hep::CalorimeterHitCollection& hcalbarrel_;
  const edm4hep::CalorimeterHitCollection& ecalendcap_;
  const edm4hep::CalorimeterHitCollection& hcalendcap_;
  const edm4hep::CalorimeterHitCollection& hcalother_;
  const edm4hep::CalorimeterHitCollection& muons_;
  const edm4hep::TrackCollection& tracks_;
  float bFieldTesla_;
};

#endif // DATAPREPROCESSING_H