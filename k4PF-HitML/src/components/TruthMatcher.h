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
#pragma once

#include "Shower.h"

#include "edm4hep/CaloHitMCParticleLinkCollection.h"
#include "edm4hep/MCParticle.h"
#include "edm4hep/RecoMCParticleLinkCollection.h"
#include "edm4hep/ReconstructedParticle.h"
#include "edm4hep/TrackMCParticleLinkCollection.h"
#include "podio/ObjectID.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numbers>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct TruthMatchConfig {
  float iouThreshold{0.25f};
  float barrelRadius{2150.f};
  int nBarrelSides{12};
  float endCapZ{2307.f};
};

// The ObjectID of an object is only used for reporting and for a deterministic ordering. It does not identify an
// object, since collections that have not (yet) been read from or written to a file all have the same (untracked)
// collection ID. Objects are identified via their handles (operator== and std::hash) instead.
struct ObjectIDKey {
  int index{-1};
  unsigned collectionID{0};

  bool valid() const { return index >= 0; }
};

inline ObjectIDKey makeKey(const podio::ObjectID& id) { return ObjectIDKey{id.index, id.collectionID}; }

inline bool objectIDLess(const podio::ObjectID& a, const podio::ObjectID& b) {
  if (a.index != b.index) {
    return a.index < b.index;
  }
  return a.collectionID < b.collectionID;
}

struct TruthLabel {
  bool valid{false};
  edm4hep::MCParticle mc{};
  ObjectIDKey key{};
};

struct ShowerTruthMatch {
  bool matched{false};
  edm4hep::MCParticle mc{};
  ObjectIDKey key{};
  float iou{0.f};
  float sharedEnergy{0.f};
};

namespace truth_detail {

constexpr int BIT_BACKSCATTER = 29;
constexpr int BIT_DECAYED_IN_TRACKER = 27;

inline bool checkBit(int value, int bit) { return ((value >> bit) & 1) == 1; }

inline bool backscatteredAndTracker(int simStatus) {
  const bool decayedInTracker = checkBit(simStatus, BIT_DECAYED_IN_TRACKER);
  const bool backscatter = checkBit(simStatus, BIT_BACKSCATTER);
  return backscatter && !decayedInTracker;
}

inline bool isProducedInCalo(const edm4hep::Vector3d& vtx, float barrelRadius, int nBarrelSides, float endCapZ) {
  bool producedInCalo = false;
  for (int i = 0; i < nBarrelSides; ++i) {
    const double angle = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(nBarrelSides);
    const double nx = std::cos(angle);
    const double ny = std::sin(angle);
    if (vtx.x * nx + vtx.y * ny > barrelRadius) {
      producedInCalo = true;
      break;
    }
  }
  if (std::abs(vtx.z) > endCapZ) {
    producedInCalo = true;
  }
  return producedInCalo;
}

inline edm4hep::MCParticle remapToCaloMother(edm4hep::MCParticle mc, const TruthMatchConfig& cfg) {
  while (mc.isAvailable()) {
    const bool producedInCalo = isProducedInCalo(mc.getVertex(), cfg.barrelRadius, cfg.nBarrelSides, cfg.endCapZ);
    const bool keepAsIs = backscatteredAndTracker(mc.getSimulatorStatus());

    if (!producedInCalo || keepAsIs) {
      break;
    }

    const auto parents = mc.getParents();
    if (parents.empty()) {
      break;
    }
    mc = parents[0];
  }
  return mc;
}

// The (immutable) handle type of the From side of a link collection, e.g. edm4hep::CalorimeterHit
template <typename LinkCollectionT>
using LinkFromT = std::decay_t<decltype(std::declval<const LinkCollectionT&>()[0].getFrom())>;

template <typename LinkCollectionT>
std::unordered_map<LinkFromT<LinkCollectionT>, TruthLabel>
buildBestTruthMap(const LinkCollectionT& links, bool remapCaloMother, const TruthMatchConfig& cfg) {
  struct BestLinkInfo {
    float weight{-std::numeric_limits<float>::infinity()};
    edm4hep::MCParticle mc{};
  };

  std::unordered_map<LinkFromT<LinkCollectionT>, BestLinkInfo> bestPerObject;

  for (const auto& link : links) {
    const auto from = link.getFrom();
    const float w = link.getWeight();

    auto it = bestPerObject.find(from);
    if (it == bestPerObject.end() || w > it->second.weight) {
      bestPerObject[from] = BestLinkInfo{w, link.getTo()};
    }
  }

  std::unordered_map<LinkFromT<LinkCollectionT>, TruthLabel> result;
  result.reserve(bestPerObject.size());

  for (const auto& [obj, best] : bestPerObject) {
    edm4hep::MCParticle mc = best.mc;
    if (remapCaloMother) {
      mc = remapToCaloMother(mc, cfg);
    }
    result[obj] = TruthLabel{mc.isAvailable(), mc, makeKey(mc.getObjectID())};
  }

  return result;
}

// Hungarian maximization by converting to minimization cost.
inline std::vector<std::pair<int, int>> hungarianMaximize(const std::vector<std::vector<float>>& score) {
  const int nRows = static_cast<int>(score.size());
  const int nCols = (nRows > 0) ? static_cast<int>(score[0].size()) : 0;
  const int n = std::max(nRows, nCols);

  if (nRows == 0 || nCols == 0) {
    return {};
  }

  float maxVal = 0.f;
  for (const auto& row : score) {
    for (float x : row) {
      maxVal = std::max(maxVal, x);
    }
  }

  std::vector<std::vector<float>> cost(n + 1, std::vector<float>(n + 1, maxVal));
  for (int i = 1; i <= nRows; ++i) {
    for (int j = 1; j <= nCols; ++j) {
      cost[i][j] = maxVal - score[i - 1][j - 1];
    }
  }

  std::vector<float> u(n + 1), v(n + 1);
  std::vector<int> p(n + 1), way(n + 1);

  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<float> minv(n + 1, std::numeric_limits<float>::infinity());
    std::vector<char> used(n + 1, false);

    do {
      used[j0] = true;
      const int i0 = p[j0];
      float delta = std::numeric_limits<float>::infinity();
      int j1 = 0;

      for (int j = 1; j <= n; ++j) {
        if (used[j]) {
          continue;
        }
        const float cur = cost[i0][j] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }

      for (int j = 0; j <= n; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);

    do {
      const int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 != 0);
  }

  std::vector<std::pair<int, int>> assignment;
  for (int j = 1; j <= n; ++j) {
    if (p[j] >= 1 && p[j] <= nRows && j >= 1 && j <= nCols) {
      assignment.emplace_back(p[j] - 1, j - 1);
    }
  }
  return assignment;
}

} // namespace truth_detail

template <typename CaloLinkCollectionT, typename TrackLinkCollectionT>
std::vector<ShowerTruthMatch>
matchShowersByIoU(const std::vector<Shower>& showers, const CaloLinkCollectionT& caloTruthLinks,
                  const TrackLinkCollectionT& trackTruthLinks, const TruthMatchConfig& cfg = {}) {
  const auto hitTruth = truth_detail::buildBestTruthMap(caloTruthLinks, true, cfg);
  const auto trackTruth = truth_detail::buildBestTruthMap(trackTruthLinks, false, cfg);

  const auto background = edm4hep::MCParticle::makeEmpty();

  std::vector<std::unordered_map<edm4hep::MCParticle, int>> intersections(showers.size());
  std::vector<std::unordered_map<edm4hep::MCParticle, float>> intersectionsE(showers.size());
  std::unordered_map<edm4hep::MCParticle, int> truthSizes;
  std::vector<int> predSizes(showers.size(), 0);

  for (size_t i = 0; i < showers.size(); ++i) {
    const auto& shower = showers[i];

    for (const auto& hit : shower.getCalorimeterHits()) {
      ++predSizes[i];
      auto it = hitTruth.find(hit);
      const auto truth = (it != hitTruth.end() && it->second.valid) ? it->second.mc : background;

      intersections[i][truth] += 1;
      intersectionsE[i][truth] += hit.getEnergy();
      truthSizes[truth] += 1;
    }

    for (const auto& trk : shower.getTracks()) {
      ++predSizes[i];
      auto it = trackTruth.find(trk);
      const auto truth = (it != trackTruth.end() && it->second.valid) ? it->second.mc : background;

      intersections[i][truth] += 1;
      truthSizes[truth] += 1;
    }
  }

  std::vector<edm4hep::MCParticle> truthKeys;
  truthKeys.reserve(truthSizes.size());
  for (const auto& [mc, _] : truthSizes) {
    if (mc.isAvailable()) {
      truthKeys.push_back(mc);
    }
  }
  // Fixed order of the truth particles (rows) for the Hungarian matching
  std::sort(truthKeys.begin(), truthKeys.end(), [](const edm4hep::MCParticle& a, const edm4hep::MCParticle& b) {
    return objectIDLess(a.getObjectID(), b.getObjectID());
  });

  if (showers.empty() || truthKeys.empty()) {
    return std::vector<ShowerTruthMatch>(showers.size());
  }

  std::vector<std::vector<float>> iou(truthKeys.size(), std::vector<float>(showers.size(), 0.f));

  for (size_t t = 0; t < truthKeys.size(); ++t) {
    const auto& truthKey = truthKeys[t];
    const int truthSize = truthSizes[truthKey];

    for (size_t s = 0; s < showers.size(); ++s) {
      const auto it = intersections[s].find(truthKey);
      const int inter = (it != intersections[s].end()) ? it->second : 0;
      const int uni = predSizes[s] + truthSize - inter;
      if (uni > 0) {
        iou[t][s] = static_cast<float>(inter) / static_cast<float>(uni);
        if (iou[t][s] < cfg.iouThreshold) {
          iou[t][s] = 0.f;
        }
      }
    }
  }

  const auto assignment = truth_detail::hungarianMaximize(iou);

  std::vector<ShowerTruthMatch> result(showers.size());

  for (const auto& [truthRow, showerCol] : assignment) {
    const float score = iou[truthRow][showerCol];
    if (score <= 0.f) {
      continue;
    }

    const auto& truthKey = truthKeys[truthRow];
    result[showerCol].matched = true;
    result[showerCol].key = makeKey(truthKey.getObjectID());
    result[showerCol].mc = truthKey;
    result[showerCol].iou = score;

    const auto eIt = intersectionsE[showerCol].find(truthKey);
    if (eIt != intersectionsE[showerCol].end()) {
      result[showerCol].sharedEnergy = eIt->second;
    }
  }

  return result;
}

inline void fillRecoTruthLink(edm4hep::RecoMCParticleLinkCollection& outLinks,
                              const edm4hep::ReconstructedParticle& reco, const ShowerTruthMatch& match) {
  if (!match.matched) {
    return;
  }

  auto link = outLinks.create();
  link.setFrom(reco);
  link.setTo(match.mc);
  link.setWeight(match.iou);
}

struct TruthRecoSummary {
  edm4hep::MCParticle mc{};
  ObjectIDKey key{};
  float recoCaloEnergy{0.f};
  int nTracks{0};
};

template <typename CaloLinkCollectionT, typename TrackLinkCollectionT>
std::vector<TruthRecoSummary> collectTruthRecoSummaries(const CaloLinkCollectionT& caloTruthLinks,
                                                        const TrackLinkCollectionT& trackTruthLinks,
                                                        const TruthMatchConfig& cfg = {}) {
  const auto hitTruth = truth_detail::buildBestTruthMap(caloTruthLinks, true, cfg);
  const auto trackTruth = truth_detail::buildBestTruthMap(trackTruthLinks, false, cfg);

  std::unordered_map<edm4hep::MCParticle, TruthRecoSummary> out;
  std::unordered_set<truth_detail::LinkFromT<CaloLinkCollectionT>> seenHits;
  std::unordered_set<truth_detail::LinkFromT<TrackLinkCollectionT>> seenTracks;

  for (const auto& link : caloTruthLinks) {
    const auto hit = link.getFrom();
    if (!seenHits.insert(hit).second) {
      continue;
    }

    const auto it = hitTruth.find(hit);
    if (it == hitTruth.end() || !it->second.valid) {
      continue;
    }

    auto& row = out[it->second.mc];
    row.key = it->second.key;
    row.mc = it->second.mc;
    row.recoCaloEnergy += hit.getEnergy();
  }

  for (const auto& link : trackTruthLinks) {
    const auto trk = link.getFrom();
    if (!seenTracks.insert(trk).second) {
      continue;
    }

    const auto it = trackTruth.find(trk);
    if (it == trackTruth.end() || !it->second.valid) {
      continue;
    }

    auto& row = out[it->second.mc];
    row.key = it->second.key;
    row.mc = it->second.mc;
    row.nTracks += 1;
  }

  std::vector<TruthRecoSummary> result;
  result.reserve(out.size());
  for (auto& [_, row] : out) {
    result.push_back(row);
  }

  std::sort(result.begin(), result.end(), [](const TruthRecoSummary& a, const TruthRecoSummary& b) {
    if (a.key.collectionID != b.key.collectionID) {
      return a.key.collectionID < b.key.collectionID;
    }
    return a.key.index < b.key.index;
  });

  return result;
}
