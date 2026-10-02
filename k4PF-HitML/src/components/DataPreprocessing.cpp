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

#include "DataPreprocessing.h"

// edm4hep imports
#include "edm4hep/CalorimeterHitCollection.h"
#include "edm4hep/ReconstructedParticleCollection.h"
#include "edm4hep/TrackCollection.h"
#include "edm4hep/TrackerHit.h"

#include "Helpers.h"
#include "Shower.h"

#include "ONNXHelper.h"
#include <torch/torch.h>

DataPreprocessing::DataPreprocessing(const edm4hep::CalorimeterHitCollection& EcalBarrel_hits,
                                     const edm4hep::CalorimeterHitCollection& HcalBarrel_hits,
                                     const edm4hep::CalorimeterHitCollection& EcalEndcap_hits,
                                     const edm4hep::CalorimeterHitCollection& HcalEndcap_hits,
                                     const edm4hep::CalorimeterHitCollection& HcalOther_hits,
                                     const edm4hep::CalorimeterHitCollection& Muon_hits,
                                     const edm4hep::TrackCollection& tracks, float bFieldTesla)
    : ecalbarrel_(EcalBarrel_hits), hcalbarrel_(HcalBarrel_hits), ecalendcap_(EcalEndcap_hits),
      hcalendcap_(HcalEndcap_hits), hcalother_(HcalOther_hits), muons_(Muon_hits), tracks_(tracks),
      bFieldTesla_(bFieldTesla) {}

PreprocessedData DataPreprocessing::extract() const {
  // std::map<std::string, std::vector<float>> features; //features used for clustering model
  PreprocessedData out;
  auto& features = out.features;
  auto& hit_mapping = out.hit_mapping;

  features["pos_hits_xyz_hits"];
  features["e_hits"];
  features["p_hits"];
  features["hit_type_feature_hit"];
  features["pos_hits_xyz_tracks"];
  features["e_tracks"];
  features["p_tracks"];
  features["hit_type_feature_track"];

  // collection of hits
  std::vector<std::pair<std::string, const edm4hep::CalorimeterHitCollection*>> hit_collections = {
      {"ECAL_BARREL", &ecalbarrel_}, {"ECAL_ENDCAP", &ecalendcap_}, {"HCAL_BARREL", &hcalbarrel_},
      {"HCAL_ENDCAP", &hcalendcap_}, {"HCAL_OTHER", &hcalother_},   {"MUON", &muons_}};

  int collectionIndex = 0;

  for (const auto& [name, hit_collection] : hit_collections) {
    int hitIndex = 0;
    for (const auto& hit : *hit_collection) {

      auto pos = hit.getPosition();
      auto energy = hit.getEnergy();

      float x = pos.x;
      float y = pos.y;
      float z = pos.z;

      int htype;
      if (name.find("ECAL") != std::string::npos) {
        htype = 2;
      } else if (name.find("HCAL") != std::string::npos) {
        htype = 3;
      } else if (name.find("MUON") != std::string::npos) {
        htype = 4;
      } else {
        htype = 5;
      }

      features["pos_hits_xyz_hits"].push_back(x);
      features["pos_hits_xyz_hits"].push_back(y);
      features["pos_hits_xyz_hits"].push_back(z);
      features["e_hits"].push_back(energy);
      features["p_hits"].push_back(0);
      features["hit_type_feature_hit"].push_back(htype);

      // include mapping
      hit_mapping.push_back({htype, collectionIndex, hitIndex});

      hitIndex += 1;
    }

    collectionIndex += 1;
  }

  // extract track information
  int trackIndex = 0;
  for (const auto& track : tracks_) {

    auto trackstate = track.getTrackStates()[0];
    float omega = trackstate.omega;
    float phi = trackstate.phi;
    float tanLambda = trackstate.tanLambda;

    float pt = 2.99792e-4f * std::abs(bFieldTesla_ / omega);
    float px = std::cos(phi) * pt;
    float py = std::sin(phi) * pt;
    float pz = tanLambda * pt;
    float p = std::sqrt(px * px + py * py + pz * pz);

    features["p_tracks"].push_back(p);
    features["e_tracks"].push_back(0);

    // also add features for trackstate at calo
    auto trackstate_calo = track.getTrackStates()[3];
    auto referencePoint_calo = trackstate_calo.referencePoint;

    float x_c = referencePoint_calo.x;
    float y_c = referencePoint_calo.y;
    float z_c = referencePoint_calo.z;

    int htype_c = 1; // vertex track state

    features["pos_hits_xyz_tracks"].push_back(x_c);
    features["pos_hits_xyz_tracks"].push_back(y_c);
    features["pos_hits_xyz_tracks"].push_back(z_c);
    features["hit_type_feature_track"].push_back(htype_c);

    hit_mapping.push_back({htype_c, collectionIndex, trackIndex});

    trackIndex += 1;
  }

  features["node_energy"] = features["e_hits"];
  features["node_energy"].insert(features["node_energy"].end(), features["e_tracks"].begin(),
                                 features["e_tracks"].end());

  features["node_p"] = features["p_hits"];
  features["node_p"].insert(features["node_p"].end(), features["p_tracks"].begin(), features["p_tracks"].end());

  features["hit_type"] = features["hit_type_feature_hit"];
  features["hit_type"].insert(features["hit_type"].end(), features["hit_type_feature_track"].begin(),
                              features["hit_type_feature_track"].end());

  return out;
}

ClusteringInputs DataPreprocessing::convertModelInputs(std::map<std::string, std::vector<float>> features) const {

  ClusteringInputs packed;

  // prepare position
  const auto& pos_hits_flat = features.at("pos_hits_xyz_hits");
  std::size_t N = pos_hits_flat.size() / 3;

  torch::Tensor pos_hits = torch::tensor(pos_hits_flat, torch::kFloat32).reshape({static_cast<long>(N), 3});

  const auto& pos_tracks_flat = features.at("pos_hits_xyz_tracks");
  std::size_t N_tracks = pos_tracks_flat.size() / 3;

  torch::Tensor pos_tracks = torch::tensor(pos_tracks_flat, torch::kFloat32).reshape({static_cast<long>(N_tracks), 3});

  const long n_nodes = static_cast<long>(N + N_tracks);

  // concatenate pos features
  torch::Tensor pos_feature = torch::cat({pos_hits, pos_tracks}, 0);

  // prepare e
  torch::Tensor hit_e = torch::from_blob(const_cast<float*>(features.at("e_hits").data()),
                                         {static_cast<long>(features.at("e_hits").size())}, torch::kFloat32)
                            .clone();

  torch::Tensor track_e = torch::from_blob(const_cast<float*>(features.at("e_tracks").data()),
                                           {static_cast<long>(features.at("e_tracks").size())}, torch::kFloat32)
                              .clone();

  torch::Tensor node_e = torch::from_blob(const_cast<float*>(features.at("node_energy").data()),
                                          {static_cast<long>(features.at("node_energy").size())}, torch::kFloat32)
                             .clone()
                             .unsqueeze(1);

  torch::Tensor node_p = torch::from_blob(const_cast<float*>(features.at("node_p").data()),
                                          {static_cast<long>(features.at("node_p").size())}, torch::kFloat32)
                             .clone()
                             .unsqueeze(1);

  // prepare p
  torch::Tensor hit_p = torch::from_blob(const_cast<float*>(features.at("p_hits").data()),
                                         {static_cast<long>(features.at("p_hits").size())}, torch::kFloat32)
                            .clone();

  torch::Tensor track_p = torch::from_blob(const_cast<float*>(features.at("p_tracks").data()),
                                           {static_cast<long>(features.at("p_tracks").size())}, torch::kFloat32)
                              .clone();

  torch::Tensor hit_type_feature =
      torch::from_blob(const_cast<float*>(features.at("hit_type").data()),
                       {static_cast<long>(features.at("hit_type").size())}, torch::kFloat32)
          .clone();

  torch::Tensor hit_type_one_hot = torch::one_hot(hit_type_feature.to(torch::kInt64), 5).to(torch::kFloat32);

  torch::Tensor h_scalar = torch::cat({pos_feature, hit_type_one_hot, node_e, node_p}, 1).to(torch::kFloat32);

  // input 0: pos_hits_xyz  [N, 3]
  {
    size_t numel = static_cast<size_t>(pos_feature.numel());
    std::vector<float> flat(numel);
    std::memcpy(flat.data(), pos_feature.data_ptr<float>(), numel * sizeof(float));
    packed.inputs.push_back(
        ONNXInput{"pos_hits_xyz", ONNXInput::Type::Float, std::vector<long>{n_nodes, 3}, std::move(flat), {}});
  }

  // input 1: hit_type [N]
  {
    std::vector<int64_t> hit_type;
    hit_type.reserve(features.at("hit_type").size());

    for (float t : features.at("hit_type")) {
      hit_type.push_back(static_cast<int64_t>(t));
    }

    packed.inputs.push_back(
        ONNXInput{"hit_type", ONNXInput::Type::Int64, std::vector<long>{n_nodes}, {}, std::move(hit_type)});
  }

  // input 2: h_scalar  [N, 10]
  {
    size_t numel = static_cast<size_t>(h_scalar.numel());
    std::vector<float> flat(numel);
    std::memcpy(flat.data(), h_scalar.data_ptr<float>(), numel * sizeof(float));
    packed.inputs.push_back(
        ONNXInput{"h_scalar", ONNXInput::Type::Float, std::vector<long>{n_nodes, 10}, std::move(flat), {}});
  }

  packed.batch_size = static_cast<unsigned long long>(n_nodes);

  return packed;
}

static std::vector<float> flatten_points(const std::vector<float>& x, const std::vector<float>& y,
                                         const std::vector<float>& z) {
  std::vector<float> out;
  out.reserve(x.size() * 3);
  for (size_t i = 0; i < x.size(); ++i) {
    out.push_back(x[i]);
    out.push_back(y[i]);
    out.push_back(z[i]);
  }
  return out;
}

// prepare the inputs for energy regression and PID, return node and global features
std::vector<PropertyInputs> DataPreprocessing::prepare_prop(std::vector<Shower> showers) const {

  // loop over showers
  std::vector<PropertyInputs> out;
  out.reserve(showers.size());

  for (auto& shower_i : showers) {

    ONNXHelper::Tensor<float> node_features;
    node_features.reserve(9);

    std::vector<float> global_features;
    global_features.reserve(16);

    // pos from calo and track
    auto [pos_x, pos_y, pos_z] = shower_i.get_pos();

    std::vector<std::vector<float>> hit_one_hot = one_hot_encode(shower_i.types_, 4); // 3-6

    auto [e_vector, p_vector] = shower_i.get_ep(bFieldTesla_);

    std::vector<float> betas = shower_i.betas_; // 9

    node_features.push_back(pos_x);
    node_features.push_back(pos_y);
    node_features.push_back(pos_z);
    // hit_one_hot: [n_hits][4]
    for (size_t k = 0; k < 4; ++k) {
      std::vector<float> one_hot_feature;
      one_hot_feature.reserve(hit_one_hot.size());

      for (size_t i = 0; i < hit_one_hot.size(); ++i) {
        one_hot_feature.push_back(hit_one_hot[i][k]);
      }

      node_features.push_back(one_hot_feature); // 3,4,5,6
    }
    node_features.push_back(e_vector);
    node_features.push_back(p_vector);
    node_features.push_back(betas);

    // include distinction charged neutral..

    // high level stuff

    float sum_e = shower_i.getCaloEnergy(shower_i.caloHits_).first;
    float muon_e = shower_i.getCaloEnergy(shower_i.muonHits_).first;
    float ecal_e = shower_i.getCaloEnergy(shower_i.ecalHits_).first;
    float hcal_e = shower_i.getCaloEnergy(shower_i.hcalHits_).first;

    float ECAL_e_fraction = ecal_e / sum_e;
    float HCAL_e_fraction = hcal_e / sum_e;

    int num_hits = shower_i.getCalorimeterHits().size();
    int num_tracks = shower_i.getTracks().size();
    int num_muon_hits = shower_i.muonHits_.size();

    int total_hits = num_hits + num_tracks; // include tracks or not?

    float track_p = shower_i.getTrackMomentum_mean(bFieldTesla_);

    float dispersion_ecal = disperion(shower_i, shower_i.ecalHits_);
    float dispersion_hcal = disperion(shower_i, shower_i.hcalHits_);

    float chi2 = std::clamp(shower_i.Chi2_mean(), -5.0f, 5.0f);

    float mean_x = mean_var(pos_x);
    float mean_y = mean_var(pos_y);
    float mean_z = mean_var(pos_z);

    float eta = calculate_eta(mean_x, mean_y, mean_z);
    float phi = calculate_phi(mean_x, mean_y);

    // add the global features

    global_features.push_back(ECAL_e_fraction);                   // 0
    global_features.push_back(HCAL_e_fraction);                   // 1
    global_features.push_back(static_cast<float>(total_hits));    // 2
    global_features.push_back(track_p);                           // 3
    global_features.push_back(dispersion_ecal);                   // 4
    global_features.push_back(dispersion_hcal);                   // 5
    global_features.push_back(sum_e);                             // 6
    global_features.push_back(static_cast<float>(num_tracks));    // 7
    global_features.push_back(chi2);                              // 8
    global_features.push_back(muon_e);                            // 9
    global_features.push_back(static_cast<float>(num_muon_hits)); // 10
    global_features.push_back(mean_x);                            // 11
    global_features.push_back(mean_y);                            // 12
    global_features.push_back(mean_z);                            // 13
    global_features.push_back(eta);                               // 14
    global_features.push_back(phi);                               // 15

    for (auto& value : global_features) {
      if (!std::isfinite(value)) {
        value = 0.0f;
      }
    }

    PropertyInputs packed;
    std::vector<float> hits_points = flatten_points(pos_x, pos_y, pos_z);
    const long n_nodes = static_cast<long>(pos_x.size());
    if (n_nodes == 0) {
      continue;
    }

    std::vector<int64_t> hit_type;
    hit_type.reserve(shower_i.types_.size());
    for (int t : shower_i.types_) {
      hit_type.push_back(static_cast<int64_t>(t - 1));
    }

    packed.inputs.push_back(ONNXInput{"hits_points", ONNXInput::Type::Float, {n_nodes, 3}, std::move(hits_points), {}});

    packed.inputs.push_back(ONNXInput{"hit_type", ONNXInput::Type::Int64, {n_nodes}, {}, std::move(hit_type)});

    packed.inputs.push_back(ONNXInput{"betas", ONNXInput::Type::Float, {n_nodes}, betas, {}});

    packed.inputs.push_back(ONNXInput{"p", ONNXInput::Type::Float, {n_nodes}, p_vector, {}});

    packed.inputs.push_back(ONNXInput{"e", ONNXInput::Type::Float, {n_nodes}, e_vector, {}});

    packed.inputs.push_back(ONNXInput{"x_global_features",
                                      ONNXInput::Type::Float,
                                      {1, static_cast<int64_t>(global_features.size())},
                                      global_features,
                                      {}});

    out.push_back(std::move(packed));
  }

  return out;
}
