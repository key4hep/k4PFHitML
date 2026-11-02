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
#include "TruthMatcher.h"

#include "edm4hep/CaloHitMCParticleLinkCollection.h"
#include "edm4hep/RecoMCParticleLinkCollection.h"
#include "edm4hep/ReconstructedParticleCollection.h"
#include "edm4hep/TrackMCParticleLinkCollection.h"
#include "k4FWCore/Transformer.h"

#include <string>
#include <vector>

struct HitPFRecoMCTruthLinker final
    : k4FWCore::Transformer<edm4hep::RecoMCParticleLinkCollection(const edm4hep::ReconstructedParticleCollection&,
                                                                  const edm4hep::CaloHitMCParticleLinkCollection&,
                                                                  const edm4hep::TrackMCParticleLinkCollection&)> {
  HitPFRecoMCTruthLinker(const std::string& name, ISvcLocator* svcLoc)
      : Transformer(name, svcLoc,
                    {KeyValues("RecoParticles", {"HitPF"}),
                     KeyValues("CalohitMCTruthLink", {"CalohitMCTruthLink"}),
                     KeyValues("TrackMCTruthLink", {"TrackMCTruthLink"})},
                     KeyValues("RecoMCTruthLink", {"HitPFMCTruthLink"})) {}

  edm4hep::RecoMCParticleLinkCollection
  operator()(const edm4hep::ReconstructedParticleCollection& recoParticles,
             const edm4hep::CaloHitMCParticleLinkCollection& caloTruthLinks,
             const edm4hep::TrackMCParticleLinkCollection& trackTruthLinks) const override {
    std::vector<Shower> showers(recoParticles.size());
    for (std::size_t i = 0; i < recoParticles.size(); ++i) {
      for (const auto& cluster : recoParticles[i].getClusters()) {
        for (const auto& hit : cluster.getHits()) {
          showers[i].addCalorimeterHit(hit);
        }
      }
      for (const auto& track : recoParticles[i].getTracks()) {
        showers[i].addTrack(track);
      }
    }

    const TruthMatchConfig truthCfg{m_truthIoUThreshold.value(), m_truthBarrelRadius.value(),
                                    m_truthNBarrelSides.value(), m_truthEndcapZ.value()};
    const auto matches = matchShowersByIoU(showers, caloTruthLinks, trackTruthLinks, truthCfg);

    auto links = edm4hep::RecoMCParticleLinkCollection{};
    for (std::size_t i = 0; i < recoParticles.size(); ++i) {
      fillRecoTruthLink(links, recoParticles[i], matches[i]);
    }
    return links;
  }

private:
  Gaudi::Property<float> m_truthIoUThreshold{this, "TruthIoUThreshold", 0.25f,
                                             "Truth matching: minimum IoU for a shower-to-MCParticle match"};
  Gaudi::Property<float> m_truthBarrelRadius{
      this, "TruthBarrelRadius", 2150.f,
      "Truth matching: detector barrel radius [mm] used to decide if a vertex is inside the calorimeter"};
  Gaudi::Property<int> m_truthNBarrelSides{
      this, "TruthNBarrelSides", 12,
      "Truth matching: number of barrel polygon sides used for the in-calorimeter check"};
  Gaudi::Property<float> m_truthEndcapZ{
      this, "TruthEndcapZ", 2307.f,
      "Truth matching: detector endcap |z| [mm] used to decide if a vertex is inside the calorimeter"};
};

DECLARE_COMPONENT(HitPFRecoMCTruthLinker)
