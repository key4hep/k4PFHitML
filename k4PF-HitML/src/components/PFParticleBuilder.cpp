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
#include "PFParticleBuilder.h"

#include "Helpers.h"
#include "Shower.h"

#include "edm4hep/Track.h"
#include "edm4hep/TrackState.h"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>

static float massFromPredictedClass(int predictedClass) {
  switch (predictedClass) {
  case 0:
    return 0.000511f; // electron
  case 1:
    return 0.139570f; // charged hadron -> pion mass
  case 2:
    return 0.939565f; // neutral hadron -> neutron mass
  case 3:
    return 0.f; // photon
  case 4:
    return 0.105658f; // muon
  default:
    return 0.f;
  }
}

static int pdgFromChargedClass(int predictedClass, int chargeSign) {
  switch (predictedClass) {
  case 0:
    return (chargeSign < 0) ? 11 : -11; // e- / e+
  case 1:
    return (chargeSign > 0) ? 211 : -211; // pi+ / pi-
  case 4:
    return (chargeSign < 0) ? 13 : -13; // mu- / mu+
  default:
    return 0;
  }
}

static int pdgFromNeutralClass(int predictedClass) {
  switch (predictedClass) {
  case 2:
    return 2112; // neutron
  case 3:
    return 22; // photon
  default:
    return 0;
  }
}

const std::vector<float>& findPIDOutput(const ONNXHelper& model, const std::vector<std::vector<float>>& outputs) {
  const auto& names = model.outputNames();

  for (size_t i = 0; i < names.size() && i < outputs.size(); ++i) {
    if (names[i].find("pid") != std::string::npos) {
      return outputs[i];
    }
  }

  throw std::runtime_error("Could not find PID output tensor");
}

PIDPrediction decodePIDLogits(const std::vector<float>& logits, const std::vector<int>& classMap) {
  if (logits.empty()) {
    throw std::runtime_error("PID logits are empty");
  }

  if (logits.size() != classMap.size()) {
    throw std::runtime_error("PID logits size does not match class map size");
  }

  const auto bestIt = std::max_element(logits.begin(), logits.end());
  const int localIndex = static_cast<int>(std::distance(logits.begin(), bestIt));

  const float maxLogit = *bestIt;
  float sumExp = 0.f;
  for (float x : logits) {
    sumExp += std::exp(x - maxLogit);
  }

  std::vector<float> scores(logits.size(), 0.f);
  if (sumExp > 0.f) {
    for (std::size_t i = 0; i < logits.size(); ++i) {
      scores[i] = std::exp(logits[i] - maxLogit) / sumExp;
    }
  }

  PIDPrediction pred;
  pred.localIndex = localIndex;
  pred.physicsClass = classMap[localIndex];
  pred.score = scores[localIndex];
  pred.scores = std::move(scores);
  return pred;
}

static const edm4hep::Track& pickBestTrack(const Shower& shower) {
  const auto& tracks = shower.getTracks();
  if (tracks.empty()) {
    throw std::runtime_error("Charged shower has no track");
  }

  size_t bestIdx = 0;
  float bestScore = std::numeric_limits<float>::max();

  for (size_t i = 0; i < tracks.size(); ++i) {
    const float ndf = tracks[i].getNdf();
    const float score = (ndf > 0.f) ? tracks[i].getChi2() / ndf : std::numeric_limits<float>::max();

    if (score < bestScore) {
      bestScore = score;
      bestIdx = i;
    }
  }
  return tracks[bestIdx];
}

static int chargeSignFromTrack(const edm4hep::TrackState& ts) {
  // Verify convention once with your sample.
  return (ts.omega > 0.f) ? +1 : -1;
}

// Energy-weighted position centroid of a set of calorimeter hits.
static edm4hep::Vector3f energyWeightedBarycenter(const std::vector<edm4hep::CalorimeterHit>& hits) {
  float sumE = 0.f, x = 0.f, y = 0.f, z = 0.f;
  for (const auto& hit : hits) {
    const auto pos = hit.getPosition();
    const float e = hit.getEnergy();
    sumE += e;
    x += pos.x * e;
    y += pos.y * e;
    z += pos.z * e;
  }
  if (sumE <= 0.f) {
    throw std::runtime_error("energyWeightedBarycenter: zero total calorimeter energy");
  }
  return {x / sumE, y / sumE, z / sumE};
}

// like Pandora (DDPfoCreator::CalculateTrackBasedReferencePoint), no-sibling branch
static edm4hep::Vector3f trackReferencePoint(const edm4hep::Track& trk) {
  const auto ts = trk.getTrackState(edm4hep::TrackState::AtIP).value();
  return {ts.D0 * std::cos(ts.phi), ts.D0 * std::sin(ts.phi), ts.Z0};
}

ParticleRecoInfo buildChargedRecoInfo(const Shower& shower, int predictedClass, float pidScore, float bFieldTesla,
                                      bool reassignLowPMuons, float muonToChargedHadronPThreshold) {
  ParticleRecoInfo out{};

  const auto& trk = pickBestTrack(shower);
  out.tracks = {trk};
  const auto ts = trk.getTrackState(edm4hep::TrackState::AtIP).value(); // matches Python's pos_pxpypz_at_vertex

  const auto p3 = momentumFromTrackState(ts, bFieldTesla);
  const float p = momentumMagnitude(p3);

  // reassign low momentum muon to CH
  int effectiveClass = predictedClass;
  if (reassignLowPMuons && predictedClass == 4 && p < muonToChargedHadronPThreshold) {
    effectiveClass = 1;
  }
  const float mass = massFromPredictedClass(effectiveClass);
  const int chargeSign = chargeSignFromTrack(ts);

  out.momentum = p3;
  out.referencePoint = trackReferencePoint(trk);
  out.energy = std::sqrt(p * p + mass * mass); // this does not correspond to python implementation (only p)
  out.pidScore = pidScore;
  out.physicsClass = effectiveClass;
  out.mass = mass;
  out.charge = static_cast<float>(chargeSign);
  out.pdg = pdgFromChargedClass(effectiveClass, chargeSign);

  return out;
}

void fillRecoParticle(edm4hep::ReconstructedParticleCollection& pfParticles,
                      edm4hep::ParticleIDCollection& pfParticleIDs, edm4hep::ClusterCollection& pfClusters,
                      const Shower& shower, const ParticleRecoInfo& recoInfo) {
  auto rp = pfParticles.create();
  auto pid = pfParticleIDs.create();

  rp.setMomentum(recoInfo.momentum);
  rp.setEnergy(recoInfo.energy);
  rp.setMass(recoInfo.mass);
  rp.setCharge(recoInfo.charge);
  rp.setReferencePoint(recoInfo.referencePoint);

  for (const auto& trk : recoInfo.tracks) {
    rp.addToTracks(trk);
  }

  rp.setGoodnessOfPID(recoInfo.pidScore);
  rp.setPDG(recoInfo.pdg);

  pid.setParticle(rp);
  pid.setLikelihood(recoInfo.pidScore);
  pid.setType(recoInfo.physicsClass);
  pid.setPDG(recoInfo.pdg);

  for (float s : recoInfo.pidScores) {
    pid.addToParameters(s);
  }

  // add clusters similar as pandora
  const auto& hits = shower.getCalorimeterHits();
  if (!hits.empty()) {
    auto cluster = pfClusters.create();
    float hitEnergySum = 0.f;
    for (const auto& hit : hits) {
      cluster.addToHits(hit);
      hitEnergySum += hit.getEnergy();
    }

    cluster.setEnergy(recoInfo.tracks.empty() ? recoInfo.energy : hitEnergySum);
    cluster.setPosition(energyWeightedBarycenter(hits));

    cluster.addToSubdetectorEnergies(shower.getCaloEnergy(shower.ecalHits_).first);
    cluster.addToSubdetectorEnergies(shower.getCaloEnergy(shower.hcalHits_).first);
    cluster.addToSubdetectorEnergies(shower.getCaloEnergy(shower.muonHits_).first);
    cluster.addToSubdetectorEnergies(0.f);
    cluster.addToSubdetectorEnergies(0.f);
    cluster.addToSubdetectorEnergies(0.f);

    rp.addToClusters(cluster);
  }
}

edm4hep::Vector3f computeNeutralDirection(const edm4hep::Vector3f& referencePoint) {

  const float norm = std::sqrt(referencePoint.x * referencePoint.x + referencePoint.y * referencePoint.y +
                               referencePoint.z * referencePoint.z);

  if (norm <= 0.f) {
    throw std::runtime_error("Neutral reference point has zero norm");
  }

  return {referencePoint.x / norm, referencePoint.y / norm, referencePoint.z / norm};
}

// only used for the direction
edm4hep::Vector3f computeNeutralReferencePoint(const Shower& shower) {
  const auto& ecalHits = shower.ecalHits_;
  const auto& hcalHits = shower.hcalHits_;

  const float nEcal = static_cast<float>(ecalHits.size());
  const float nHcal = static_cast<float>(hcalHits.size());
  const float denom = nEcal + nHcal;

  // Match Python:
  // mask_ecal_only = (n_ecal_hits / (n_hcal_hits + n_ecal_hits)) > 0.05
  const bool useEcalOnly = (denom > 0.f) ? ((nEcal / denom) > 0.05f) : false;

  return useEcalOnly ? energyWeightedBarycenter(ecalHits) : energyWeightedBarycenter(shower.getCalorimeterHits());
}

// Pandora-like neutral reference point: energy-weighted barycenter of all cluster hits.
edm4hep::Vector3f computeNeutralPandoraReferencePoint(const Shower& shower) {
  return energyWeightedBarycenter(shower.getCalorimeterHits());
}

ParticleRecoInfo buildNeutralRecoInfo(const Shower&, int predictedClass, float pidScore, float predictedEnergy,
                                      const edm4hep::Vector3f& predictedDirection,
                                      const edm4hep::Vector3f& predictedReferencePoint) {
  ParticleRecoInfo out{};

  const float mass = massFromPredictedClass(predictedClass);

  const float pMag = std::sqrt(std::max(0.f, predictedEnergy * predictedEnergy - mass * mass));

  out.momentum = {predictedDirection.x * pMag, predictedDirection.y * pMag, predictedDirection.z * pMag};
  out.referencePoint = predictedReferencePoint;
  out.direction = predictedDirection;

  out.energy = predictedEnergy;
  out.pidScore = pidScore;
  out.physicsClass = predictedClass;
  out.mass = mass;
  out.pdg = pdgFromNeutralClass(predictedClass);

  return out;
}
