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

#include "edm4hep/MCParticleCollection.h"
#include "edm4hep/ParticleIDCollection.h"
#include "k4FWCore/Transformer.h"
#include <string>

// edm4hep imports
#include "edm4hep/CaloHitMCParticleLinkCollection.h"
#include "edm4hep/CalorimeterHitCollection.h"
#include "edm4hep/RecoMCParticleLinkCollection.h"
#include "edm4hep/ReconstructedParticleCollection.h"
#include "edm4hep/TrackCollection.h"
#include "edm4hep/TrackMCParticleLinkCollection.h"
#include "edm4hep/TrackerHit.h"

// others
#include "Clustering.h"
#include "DataPreprocessing.h"
#include "EvaluationSummary.h"
#include "Helpers.h"
#include "PFParticleBuilder.h"
#include "ROOT/RVec.hxx"
#include "Shower.h"
#include "ShowerBuilder.h"
#include "TruthMatcher.h"
#include <fstream>
#include <iomanip>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <unordered_map>
#include <unordered_set>

// ONNX
#include "ONNXHelper.h"
#include "onnxruntime_cxx_api.h"
#include <torch/torch.h>

using RecoTruthLinkCollection = edm4hep::RecoMCParticleLinkCollection;
using CaloTruthLinkCollection = edm4hep::CaloHitMCParticleLinkCollection;
using TrackTruthLinkCollection = edm4hep::TrackMCParticleLinkCollection;

namespace rv = ROOT::VecOps;

namespace {

// Validation dump: raw clustering-model inputs, one row per hit/track node,
// in the exact order they were fed to ONNX (pos_hits_xyz, hit_type, h_scalar).
void dumpClusteringInputs(const ClusteringInputs& clustering_input, std::size_t eventIdx) {
  const auto& pos = clustering_input.inputs.at(0).float_data;      // pos_hits_xyz [N,3]
  const auto& hit_type = clustering_input.inputs.at(1).int64_data; // hit_type [N]
  const auto& h_scalar = clustering_input.inputs.at(2).float_data; // h_scalar [N,10]
  const std::size_t n = static_cast<std::size_t>(clustering_input.batch_size);

  std::ofstream out("dump/cpp_event_" + std::to_string(eventIdx) + "_input.txt");
  out << std::setprecision(9);
  out << n << " 14\n"; // columns: x y z hit_type h_scalar[0..9]
  for (std::size_t i = 0; i < n; ++i) {
    out << pos[3 * i + 0] << " " << pos[3 * i + 1] << " " << pos[3 * i + 2] << " " << hit_type[i];
    for (int c = 0; c < 10; ++c) {
      out << " " << h_scalar[10 * i + static_cast<std::size_t>(c)];
    }
    out << "\n";
  }
}

// Validation dump: raw clustering-model output (x, y, z, beta) per node.
void dumpClusteringOutput(const std::vector<float>& out0, std::size_t eventIdx) {
  const std::size_t n_rows = out0.size() / 4;
  std::ofstream out("dump/cpp_event_" + std::to_string(eventIdx) + "_output.txt");
  out << std::setprecision(9);
  out << n_rows << " 4\n";
  for (std::size_t i = 0; i < n_rows; ++i) {
    out << out0[4 * i + 0] << " " << out0[4 * i + 1] << " " << out0[4 * i + 2] << " " << out0[4 * i + 3] << "\n";
  }
}

// Validation dump: DPC cluster labels after remove_bad_tracks_from_cluster
// (the pre-correction labels/rho/delta/nearest/id are dumped separately by
// Clustering::get_clustering itself, into *_dpc.txt).
void dumpClusterLabelsPost(const torch::Tensor& labels, std::size_t eventIdx) {
  const auto labelsAcc = labels.accessor<int64_t, 1>();
  const std::size_t n = static_cast<std::size_t>(labels.size(0));
  std::ofstream out("dump/cpp_event_" + std::to_string(eventIdx) + "_dpc_label_post.txt");
  out << n << " 1\n";
  for (std::size_t i = 0; i < n; ++i) {
    out << labelsAcc[static_cast<int64_t>(i)] << "\n";
  }
}

void dumpShowerRegressionInputs(const std::vector<Shower>& showers, const std::vector<PropertyInputs>& prop_inputs,
                                std::size_t eventIdx) {
  auto findInput = [](const std::vector<ONNXInput>& inputs, const std::string& name) -> const ONNXInput& {
    for (const auto& in : inputs) {
      if (in.name == name)
        return in;
    }
    throw std::runtime_error("dumpShowerRegressionInputs: missing input " + name);
  };

  for (std::size_t i = 0; i < showers.size() && i < prop_inputs.size(); ++i) {
    const int64_t label = showers[i].label_;
    const auto& inputs = prop_inputs[i].inputs;

    const auto& hitsPoints = findInput(inputs, "hits_points").float_data;   // [N,3]
    const auto& hitType = findInput(inputs, "hit_type").int64_data;         // [N]
    const auto& p = findInput(inputs, "p").float_data;                      // [N]
    const auto& e = findInput(inputs, "e").float_data;                      // [N]
    const auto& global = findInput(inputs, "x_global_features").float_data; // [16]

    const std::size_t n = hitType.size();

    {
      std::ofstream out("dump/cpp_event_" + std::to_string(eventIdx) + "_shower_" + std::to_string(label) +
                        "_hits.txt");
      out << std::setprecision(9);
      out << n << " 6\n"; // columns: x y z hit_type p e
      for (std::size_t k = 0; k < n; ++k) {
        out << hitsPoints[3 * k + 0] << " " << hitsPoints[3 * k + 1] << " " << hitsPoints[3 * k + 2] << " "
            << hitType[k] << " " << p[k] << " " << e[k] << "\n";
      }
    }
    {
      std::ofstream out("dump/cpp_event_" + std::to_string(eventIdx) + "_shower_" + std::to_string(label) +
                        "_global.txt");
      out << std::setprecision(9);
      out << 1 << " " << global.size() << "\n";
      for (std::size_t k = 0; k < global.size(); ++k) {
        out << (k ? " " : "") << global[k];
      }
      out << "\n";
    }
  }
}

void dumpChargedRefPointDebug(const Shower& shower, int64_t label, std::size_t eventIdx) {
  const auto& tracks = shower.getTracks();
  if (tracks.empty())
    return;

  std::size_t bestIdx = 0;
  float bestScore = std::numeric_limits<float>::max();
  for (std::size_t i = 0; i < tracks.size(); ++i) {
    const float ndf = tracks[i].getNdf();
    const float score = (ndf > 0.f) ? tracks[i].getChi2() / ndf : std::numeric_limits<float>::max();
    if (score < bestScore) {
      bestScore = score;
      bestIdx = i;
    }
  }
  const auto caloState = tracks[bestIdx].getTrackStates()[3]; // AtCalorimeter
  const auto& p_xyz = caloState.referencePoint;

  float sumE = 0.f, x = 0.f, y = 0.f, z = 0.f;
  for (const auto& hit : shower.getCalorimeterHits()) {
    const auto pos = hit.getPosition();
    const float e = hit.getEnergy();
    sumE += e;
    x += pos.x * e;
    y += pos.y * e;
    z += pos.z * e;
  }
  const float bx = (sumE > 0.f) ? x / sumE : 0.f;
  const float by = (sumE > 0.f) ? y / sumE : 0.f;
  const float bz = (sumE > 0.f) ? z / sumE : 0.f;

  std::ofstream out("dump/cpp_event_" + std::to_string(eventIdx) + "_shower_" + std::to_string(label) +
                    "_charged_refpt_debug.txt");
  out << std::setprecision(9);
  out << 1 << " 9\n"; // columns: p_xyz_calo(3) barycenter(3) offset=barycenter-p_xyz_calo(3)
  out << p_xyz.x << " " << p_xyz.y << " " << p_xyz.z << " " << bx << " " << by << " " << bz << " " << (bx - p_xyz.x)
      << " " << (by - p_xyz.y) << " " << (bz - p_xyz.z) << "\n";
}

void dumpShowerRegressionOutput(const std::vector<float>& values, int64_t label, const std::string& suffix,
                                std::size_t eventIdx) {
  std::ofstream out("dump/cpp_event_" + std::to_string(eventIdx) + "_shower_" + std::to_string(label) + "_" + suffix +
                    ".txt");
  out << std::setprecision(9);
  out << 1 << " " << values.size() << "\n";
  for (std::size_t k = 0; k < values.size(); ++k) {
    out << (k ? " " : "") << values[k];
  }
  out << "\n";
}

} // namespace

struct PFHitML final
    : k4FWCore::MultiTransformer<std::tuple<edm4hep::ReconstructedParticleCollection, edm4hep::ParticleIDCollection,
                                            RecoTruthLinkCollection, edm4hep::TrackCollection>(

          const edm4hep::CalorimeterHitCollection&, const edm4hep::CalorimeterHitCollection&,
          const edm4hep::CalorimeterHitCollection&, const edm4hep::CalorimeterHitCollection&,
          const edm4hep::CalorimeterHitCollection&, const edm4hep::CalorimeterHitCollection&,
          const edm4hep::TrackCollection&, const edm4hep::MCParticleCollection&, const CaloTruthLinkCollection&,
          const TrackTruthLinkCollection&

          )> {

  PFHitML(const std::string& name, ISvcLocator* svcLoc)
      : k4FWCore::MultiTransformer<std::tuple<edm4hep::ReconstructedParticleCollection, edm4hep::ParticleIDCollection,
                                              RecoTruthLinkCollection, edm4hep::TrackCollection>(
            const edm4hep::CalorimeterHitCollection&, const edm4hep::CalorimeterHitCollection&,
            const edm4hep::CalorimeterHitCollection&, const edm4hep::CalorimeterHitCollection&,
            const edm4hep::CalorimeterHitCollection&, const edm4hep::CalorimeterHitCollection&,
            const edm4hep::TrackCollection&, const edm4hep::MCParticleCollection&, const CaloTruthLinkCollection&,
            const TrackTruthLinkCollection&)>(
            name, svcLoc,
            {KeyValues("EcalBarrelHits", {"ECALBarrel"}), KeyValues("HcalBarrelHits", {"HCALBarrel"}),
             KeyValues("EcalEndcaplHits", {"ECALEndcap"}), KeyValues("HcalEndcapHits", {"HCALEndcap"}),
             KeyValues("HcalOtherHits", {"HCALOther"}), KeyValues("MUON", {"MUON"}),
             KeyValues("Tracks", {"SiTracks_Refitted"}), KeyValues("MCParticles", {"MCParticles"}),
             KeyValues("CalohitMCTruthLink", {"CalohitMCTruthLink"}),
             KeyValues("SiTracksMCTruthLink", {"SiTracksMCTruthLink"})},
            {KeyValues("HitPF", {"HitPF"}), KeyValues("HitPFIDs", {"HitPFIDs"}), // Outputs
             KeyValues("HitPFMCTruthLink", {"HitPFMCTruthLink"}),
             KeyValues("HitPFUnassociatedTracks", {"HitPFUnassociatedTracks"})}) {}

  // main
  std::tuple<edm4hep::ReconstructedParticleCollection, edm4hep::ParticleIDCollection, RecoTruthLinkCollection,
             edm4hep::TrackCollection>
  operator()(const edm4hep::CalorimeterHitCollection& EcalBarrel_hits,
             const edm4hep::CalorimeterHitCollection& HcalBarrel_hits,
             const edm4hep::CalorimeterHitCollection& EcalEndcap_hits,
             const edm4hep::CalorimeterHitCollection& HcalEndcap_hits,
             const edm4hep::CalorimeterHitCollection& HcalOther_hits,
             const edm4hep::CalorimeterHitCollection& Muon_hits, const edm4hep::TrackCollection& tracks,
             const edm4hep::MCParticleCollection&, const CaloTruthLinkCollection& caloTruthLinks,
             const TrackTruthLinkCollection& trackTruthLinks) const override {

    debug() << "EcalBarrelHits: " << EcalBarrel_hits.size() << endmsg;
    debug() << "HcalBarrelHits: " << HcalBarrel_hits.size() << endmsg;
    debug() << "EcalEndcapHits: " << EcalEndcap_hits.size() << endmsg;
    debug() << "HcalEndcapHits: " << HcalEndcap_hits.size() << endmsg;
    debug() << "HcalOtherHits: " << HcalOther_hits.size() << endmsg;
    debug() << "MuonHits: " << Muon_hits.size() << endmsg;
    debug() << "tracks: " << tracks.size() << endmsg;

    DataPreprocessing extractor(EcalBarrel_hits, HcalBarrel_hits, EcalEndcap_hits, HcalEndcap_hits, HcalOther_hits,
                                Muon_hits, tracks, bFieldTesla.value());

    // get input variables
    auto inputs = extractor.extract();
    auto inputs_features = inputs.features;
    // convert inputs to expected shape
    auto clustering_input = extractor.convertModelInputs(inputs_features);

    if (m_eventCounter < m_maxDumpEvents) {
      dumpClusteringInputs(clustering_input, m_eventCounter);
    }

    ///////////////////////////////////////////////////
    ////////// Inference Clustering Model //////////
    ///////////////////////////////////////////////////

    info() << "================ EVENT " << m_eventCounter << " ================" << endmsg;

    std::vector<std::vector<float>> outputs = m_onnx->runNamed(clustering_input.inputs);

    // get two outputs, first one has shape (N,4) (three coordinates in embedding space + beta)
    //  second output is  dummy for pred_energy_corr (not needed at this stage)

    if (m_eventCounter < m_maxDumpEvents) {
      dumpClusteringOutput(outputs.at(0), m_eventCounter);
    }

    if (clustering_input.batch_size == 0) {
      info() << "Event " << m_eventCounter << ": no calorimeter hits or tracks, skipping" << endmsg;
      ++m_eventCounter;
      auto unassoc = edm4hep::TrackCollection{};
      unassoc.setSubsetCollection();
      return {edm4hep::ReconstructedParticleCollection{}, edm4hep::ParticleIDCollection{},
              RecoTruthLinkCollection{}, std::move(unassoc)};
    }

    /////////////////////////////////////
    ////////// CLUSTERING STEP //////////
    /////////////////////////////////////

    // expect output tensor of length N with cluster labels

    Clustering clusterer(dpc_d_c.value(), dpc_rho_min.value(), dpc_delta_min.value(), dpc_core_radius.value());
    const int64_t dpcDumpIdx = (m_eventCounter < m_maxDumpEvents) ? static_cast<int64_t>(m_eventCounter) : -1;
    torch::Tensor cluster_label = clusterer.get_clustering(outputs[0], inputs_features["node_energy"],
                                                           dpcDumpIdx); // length of hits [000 3 7 7 7 7 10 2 2 2 ..]

    // cluster postprocessing
    torch::Tensor cluster_label_corrected = clusterer.remove_bad_tracks_from_cluster(
        cluster_label, inputs_features["hit_type"], inputs_features["node_energy"], inputs_features["node_p"]);

    if (m_eventCounter < m_maxDumpEvents) {
      dumpClusterLabelsPost(cluster_label_corrected, m_eventCounter);
    }

    // this function gets the clusters, within this function create shower instances, set x,y,z,..
    ShowerBuilder builder(extractor, inputs);
    auto showers = builder.buildShowers(cluster_label_corrected, outputs[0]);

    auto split = splitShowersByTrackContent(showers);

    debug() << "Number of charged showers: " << split.charged.size() << endmsg;
    debug() << "Number of neutral showers: " << split.neutral.size() << endmsg;

    // truth linking
    TruthMatchConfig truthCfg;
    truthCfg.iouThreshold = truth_iou_threshold.value();
    truthCfg.barrelRadius = truth_barrel_radius.value();
    truthCfg.nBarrelSides = truth_n_barrel_sides.value();
    truthCfg.endCapZ = truth_endcap_z.value();

    const auto showerTruthMatches = matchShowersByIoU(showers, caloTruthLinks, trackTruthLinks, truthCfg);

    ////////////////////////////////////
    //// ENERGY REGRESSION & PID ///////
    ////////////////////////////////////

    // look here: https://github.com/selvaggi/mlpf/blob/main/src/utils/post_clustering_features.py
    auto prop_inputs = extractor.prepare_prop(showers); // determine and convert inputs for regression model

    if (m_eventCounter < m_maxDumpEvents) {
      dumpShowerRegressionInputs(showers, prop_inputs, m_eventCounter);
    }

    //////////////////////////////////////////////////
    ////////// Inference Property Model //////////////
    //////////////////////////////////////////////////

    const std::vector<int> chargedClassMap = {0, 1, 4};
    const std::vector<int> neutralClassMap = {2, 3};

    auto HitPF = edm4hep::ReconstructedParticleCollection{};
    auto HitPFIDs = edm4hep::ParticleIDCollection{};
    auto HitPFMCTruthLink = RecoTruthLinkCollection{};

    const auto truthSummaries = collectTruthRecoSummaries(caloTruthLinks, trackTruthLinks, truthCfg);

    EvaluationSummaryBuilder evalBuilder(showers, showerTruthMatches, truthSummaries);

    // loop over showers per event
    for (size_t idx : split.charged) {

      auto prop_outputs = m_onnx_prop_charged->runNamed(prop_inputs[idx].inputs);
      const auto& pidLogits = findPIDOutput(*m_onnx_prop_charged, prop_outputs);
      PIDPrediction pid = decodePIDLogits(pidLogits, chargedClassMap);

      if (m_eventCounter < m_maxDumpEvents) {
        dumpShowerRegressionOutput(pidLogits, showers[idx].label_, "charged_output", m_eventCounter);
        dumpChargedRefPointDebug(showers[idx], showers[idx].label_, m_eventCounter);
      }

      ParticleRecoInfo recoInfo =
          buildChargedRecoInfo(showers[idx], pid.physicsClass, pid.score, bFieldTesla.value(),
                               reassign_low_p_muons.value(), muon_to_charged_hadron_p_threshold.value());
      recoInfo.pidScores = pid.scores;

      evalBuilder.addRecoResult(idx, recoInfo);

      const auto recoIndex = HitPF.size();
      fillRecoParticle(HitPF, HitPFIDs, showers[idx], recoInfo);

      const auto reco = HitPF.at(recoIndex);
      fillRecoTruthLink(HitPFMCTruthLink, reco, showerTruthMatches[idx]);

      const auto pidObj = HitPFIDs.at(recoIndex);

      debug() << "=== C++ final charged properties === "
              << "idx=" << idx << " calibrated_E=" << reco.getEnergy() << " momentum=" << recoInfo.momentum
              << " mass=" << recoInfo.mass << " pid_class=" << pidObj.getType()
              << " pid_likelihood=" << pidObj.getLikelihood() << endmsg;
    }

    for (size_t idx : split.neutral) {

      auto prop_outputs = m_onnx_prop_neutral->runNamed(prop_inputs[idx].inputs);
      const auto& pidLogits = findPIDOutput(*m_onnx_prop_neutral, prop_outputs);
      PIDPrediction pid = decodePIDLogits(pidLogits, neutralClassMap);
      float predictedEnergy = prop_outputs[0].at(0);
      edm4hep::Vector3f predictedReferencePoint = computeNeutralReferencePoint(showers[idx]);
      edm4hep::Vector3f predictedDirection = computeNeutralDirection(predictedReferencePoint);

      if (m_eventCounter < m_maxDumpEvents) {
        std::vector<float> combined{predictedEnergy};
        combined.insert(combined.end(), pidLogits.begin(), pidLogits.end());
        dumpShowerRegressionOutput(combined, showers[idx].label_, "neutral_output", m_eventCounter);
      }

      ParticleRecoInfo recoInfo = buildNeutralRecoInfo(showers[idx], pid.physicsClass, pid.score, predictedEnergy,
                                                       predictedDirection, predictedReferencePoint);
      recoInfo.pidScores = pid.scores;

      evalBuilder.addRecoResult(idx, recoInfo);

      const auto recoIndex = HitPF.size();
      fillRecoParticle(HitPF, HitPFIDs, showers[idx], recoInfo);

      const auto reco = HitPF.at(recoIndex);
      fillRecoTruthLink(HitPFMCTruthLink, reco, showerTruthMatches[idx]);
    }

    const auto evalRows = evalBuilder.finalize();

    // Tracks that never ended up in any shower written out as a subset collection for downstream recovery, e.g.
    // invariant mass calculations.
    auto HitPFUnassociatedTracks = edm4hep::TrackCollection{};
    HitPFUnassociatedTracks.setSubsetCollection();
    if (writeUnassociatedTracks.value()) {
      std::unordered_set<ObjectIDKey, ObjectIDKeyHash> usedTrackKeys;
      for (const auto& shower_i : showers) {
        for (const auto& trk : shower_i.getTracks()) {
          usedTrackKeys.insert(makeKey(trk.getObjectID()));
        }
      }
      for (const auto& trk : tracks) {
        if (usedTrackKeys.find(makeKey(trk.getObjectID())) == usedTrackKeys.end()) {
          HitPFUnassociatedTracks.push_back(trk);
        }
      }
      info() << "Unassociated tracks: " << HitPFUnassociatedTracks.size() << " / " << tracks.size() << endmsg;
    }

    ++m_eventCounter;

    return {std::move(HitPF), std::move(HitPFIDs), std::move(HitPFMCTruthLink),
            std::move(HitPFUnassociatedTracks)}; // outputs
  }

  StatusCode initialize() override {
    info() << "Initializing PFHitML and loading model..." << endmsg;

    m_onnx = std::make_unique<ONNXHelper>(model_path_clustering.value());

    m_onnx_prop_neutral = std::make_unique<ONNXHelper>(model_path_properties_neutral.value());

    m_onnx_prop_charged = std::make_unique<ONNXHelper>(model_path_properties_charged.value());

    return StatusCode::SUCCESS;
  }

private:
  std::unique_ptr<ONNXHelper> m_onnx;
  std::unique_ptr<ONNXHelper> m_onnx_prop_neutral;
  std::unique_ptr<ONNXHelper> m_onnx_prop_charged;
  mutable std::size_t m_eventCounter{0}; // for debugging

  rv::RVec<std::string> vars;

  Gaudi::Property<std::string> model_path_clustering{this, "model_path_clustering",
                                                     "/eos/user/l/lherrman/FCC/models/clustering_truth_update.onnx",
                                                     "Path to the ONNX clustering model"};

  Gaudi::Property<std::string> model_path_properties_neutral{
      this, "model_path_properties_neutral", "/eos/user/l/lherrman/FCC/models/energy_correction_paper_neutral.onnx",
      "Path to the ONNX model for neutral energy regression and PID"};

  Gaudi::Property<std::string> model_path_properties_charged{
      this, "model_path_properties_charged", "/eos/user/l/lherrman/FCC/models/energy_correction_paper_charged_pid.onnx",
      "Path to the ONNX model for charged energy regression and PID"};

  Gaudi::Property<std::size_t> m_maxDumpEvents{
      this, "maxDumpEvents", 0,
      "Number of leading events to dump validation tensors for into dump/ (0 disables dumping)"};

  Gaudi::Property<float> dpc_d_c{this, "dpc_d_c", 0.1f, "DPC clustering: Gaussian kernel bandwidth for local density"};

  Gaudi::Property<float> dpc_rho_min{this, "dpc_rho_min", 0.05f,
                                     "DPC clustering: minimum local density for a point to be a cluster center"};

  Gaudi::Property<float> dpc_delta_min{
      this, "dpc_delta_min", 0.4f,
      "DPC clustering: minimum distance-to-higher-density for a point to be a cluster center"};

  Gaudi::Property<float> dpc_core_radius{
      this, "dpc_core_radius", 0.5f,
      "DPC clustering: max distance to a center for a hit to be kept as a core (non-halo) member"};

  Gaudi::Property<float> truth_iou_threshold{this, "truth_iou_threshold", 0.25f,
                                             "Truth matching: minimum IoU for a shower-to-MCParticle match"};

  Gaudi::Property<float> truth_barrel_radius{
      this, "truth_barrel_radius", 2150.f,
      "Truth matching: detector barrel radius [mm] used to decide if a vertex is inside the calorimeter"};

  Gaudi::Property<int> truth_n_barrel_sides{
      this, "truth_n_barrel_sides", 12,
      "Truth matching: number of barrel polygon sides used for the in-calorimeter check"};

  Gaudi::Property<float> truth_endcap_z{
      this, "truth_endcap_z", 2307.f,
      "Truth matching: detector endcap |z| [mm] used to decide if a vertex is inside the calorimeter"};

  Gaudi::Property<float> bFieldTesla{
      this, "bFieldTesla", 2.0f,
      "Magnetic field strength [T] used for track pT reconstruction from track curvature (omega)"};

  Gaudi::Property<bool> reassign_low_p_muons{this, "reassign_low_p_muons", true,
                                             "Reassign charged candidates predicted as muon with momentum below "
                                             "muon_to_charged_hadron_p_threshold to charged hadron"};

  Gaudi::Property<float> muon_to_charged_hadron_p_threshold{
      this, "muon_to_charged_hadron_p_threshold", 1.0f,
      "Momentum threshold [GeV] below which a predicted muon is reassigned to charged hadron (if reassign_low_p_muons "
      "is set)"};

  Gaudi::Property<bool> writeUnassociatedTracks{
      this, "writeUnassociatedTracks", false,
      "Write tracks that were not assigned to any shower (DPC noise or failed the E/p consistency "
      "check) into the HitPFUnassociatedTracks output collection"};
};

DECLARE_COMPONENT(PFHitML)
