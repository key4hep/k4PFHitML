# k4PFHitML

`k4PFHitML` is the key4hep/Gaudi inference implementation of the hit-based
particle-flow algorithm described in
[arXiv:2603.04084](https://arxiv.org/html/2603.04084v1). The standalone Python
reference implementation used for training (and for inference validation
against this package) lives at
[github.com/doloresgarcia/HitPF](https://github.com/doloresgarcia/HitPF).

The package handles data preparation from EDM4hep input collections, runs the
trained models (converted to ONNX) via `k4FWCore::MultiTransformer`, and
writes out particle-flow objects as standard EDM4hep collections.

This package targets the **CLD** detector concept and geometry: input
collection names (`ECALBarrel`, `ECALEndcap`, `HCALBarrel`, `HCALEndcap`,
`HCALOther`, `MUON`, `SiTracks_Refitted`, see the `KeyValues` in
`PFHitML.cpp`), the truth-matching barrel/endcap geometry constants
(`truth_barrel_radius`, `truth_n_barrel_sides`, `truth_endcap_z`), and the
models themselves.

## Pipeline overview

Reconstruction proceeds in two model stages per event:

1. **Condensation/clustering model** -- embeds every calorimeter hit and
   track into a learned space and predicts, per hit/track, a condensation
   point/beta value. Density Peak Clustering (DPC) on that embedding groups
   hits and tracks belonging to the same particle into a `Shower`
   (`Clustering.h/.cpp`, `ShowerBuilder.h/.cpp`). The DPC algorithm itself is
   a  reimplementation of
   [pgoltstein/densitypeakclustering](https://github.com/pgoltstein/densitypeakclustering)
   (ρ local density, δ distance-to-higher-density, core/halo cluster
   assignment).
2. **Regression/PID model** -- per shower (charged and neutral run through
   separate models), regresses energy and classifies particle type
   (electron / charged hadron / neutral hadron / photon / muon)
   (`PFParticleBuilder.h/.cpp`).

Training data/samples:
- Both models are trained on CLD full simulation.
- The condensation model is trained on a Z→uds sample.
- The regression model is trained on a jet-like particle gun sample.

**Truth matching** is done via IoU (intersection-over-union) between a
reconstructed shower's constituent hits/tracks and each MCParticle's,
resolved with Hungarian matching (`TruthMatcher.h`). This produces the
`HitPFMCTruthLink` output collection, and is what any downstream fake-rate /
efficiency definition should be built on (a reconstructed particle with no
truth match is a fake; a truth particle with no reconstructed match is
missed).

## Outputs

- `HitPF` (`ReconstructedParticleCollection`): the reconstructed particle-flow
  objects -- momentum, energy, mass, charge, PDG, reference point.
- `HitPFIDs` (`ParticleIDCollection`): PID hypothesis per particle (type,
  likelihood, PDG, full per-class score vector).
- `HitPFMCTruthLink` (`RecoMCParticleLinkCollection`): reco-to-truth links
  from the IoU matching above, weight = IoU.
- `HitPFUnassociatedTracks` (`TrackCollection`, opt-in): tracks that never
  ended up in any shower (DPC noise, or removed by the E/p consistency cut),
  so never made it onto any `HitPF` particle -- see below.

## Dependencies

* ROOT

* PODIO

* Gaudi

* k4FWCore

## Installation

Run, from the `k4PFHitML` directory:

``` bash
cd /your/path/to/this/repo/k4PFHitML/
source ./setup.sh
k4_local_repo
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install -G Ninja -DPython_EXECUTABLE=$(which python3)
ninja install
```

Alternatively you can source the nightlies instead of the releases:

``` bash
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
```

## Getting the models

The three trained models are published at
<https://key4hep.web.cern.ch/testFiles/k4PFHitML/>. Download them once into a
`models/` directory at the root of this repository (about 120 MB in total),
which is where `PerformMLPF.py` looks for them by default:

``` bash
mkdir -p models
cd models
for model in clustering_paper.onnx \
             energy_correction_paper_neutral.onnx \
             energy_correction_paper_charged_pid.onnx; do
  curl -LO "https://key4hep.web.cern.ch/testFiles/k4PFHitML/${model}"
done
```

## Input files

The pipeline runs on a **reconstructed** EDM4hep file from CLD full simulation
-- the output of `CLDReconstruction.py` from
[CLDConfig](https://github.com/key4hep/CLDConfig), conventionally named
`*_REC.edm4hep.root`. The raw `ddsim` output is *not* enough: the collections
`PFHitML` reads only exist after digitisation and reconstruction.

Such a file must contain:

- the calorimeter hit collections `ECALBarrel`, `ECALEndcap`, `HCALBarrel`,
  `HCALEndcap`, `HCALOther` and `MUON`, and
- the track collection `SiTracks_Refitted`.


A ready-made example file to try the pipeline on (Z->ss):

``` bash
curl -LO "https://cernbox.cern.ch/remote.php/dav/public-files/162pSTet8hUf4xD/out_reco_edm4hep_REC.edm4hep.root"
```



## Execute

Run from the root of the repository:

``` bash
k4run k4PF-HitML/options/PerformMLPF.py --inputFiles <input.edm4hep.root> --outputFile output_HitPF.edm4hep.root
```


`PerformMLPF.py` also accepts all the configurable parameters listed below
(`--dpc_d_c`, `--bFieldTesla`, `--write_unassociated_tracks`, etc. --
run with `--help` for the full list).

The output is a ROOT file containing the input collections (`keep *`) plus
the `HitPF`/`HitPFIDs`/`HitPFMCTruthLink` (and, if enabled,
`HitPFUnassociatedTracks`) collections described above.

## Source files

- `PFHitML.cpp` -- the main Gaudi `MultiTransformer` algorithm; orchestrates
  the full per-event pipeline (data prep -> clustering ONNX -> DPC -> shower
  building -> regression ONNX -> particle assembly) and declares all
  configurable `Gaudi::Property`s.
- `DataPreprocessing.h/.cpp` -- builds the clustering model's input graph
  (calorimeter hits and tracks as nodes), and prepares per-shower inputs for
  the regression model.
- `Clustering.h/.cpp` -- Density Peak Clustering (DPC) on the clustering
  model's embedding output, plus the post-clustering E/p consistency cut
  (`remove_bad_tracks_from_cluster`) that can evict a track from its cluster.
- `ShowerBuilder.h/.cpp` -- turns DPC cluster labels into `Shower` objects,
  assigning each hit/track node back to its shower.
- `Shower.h/.cpp` -- per-shower data container (hits, tracks, betas) and
  shower-level derived quantities (mean track momentum, chi2, energy/momentum
  sums).
- `PFParticleBuilder.h/.cpp` -- turns a `Shower` plus regression/PID output
  into a final `HitPF`/`HitPFIDs` entry (`fillRecoParticle`), including all
  the charged/neutral-specific conventions documented below.
- `TruthMatcher.h` -- IoU-based Hungarian matching between reconstructed
  showers and MCParticles; produces `HitPFMCTruthLink`.
- `EvaluationSummary.h/.cpp` -- collects per-event reco/truth summary rows
  (used for validation/debug logging, not part of the physics output).
- `ONNXHelper.h/.cpp` -- thin wrapper around ONNX Runtime for loading a model
  and running named inference.
- `Helpers.h/.cpp` -- shared small utility functions used across the above.

## HitPF configurable parameters

All exposed as `Gaudi::Property`s on `PFHitML`, settable from
`options/PerformMLPF.py`:

- The three ONNX model paths (`model_path_clustering`,
  `model_path_properties_neutral`, `model_path_properties_charged` --
  `--onnx_model_clustering`/`--onnx_model_properties_neutral`/
  `--onnx_model_properties_charged` on the command line).
- Density Peak Clustering parameters (`dpc_d_c`, `dpc_rho_min`,
  `dpc_delta_min`, `dpc_core_radius`).
- Truth matching parameters (`truth_iou_threshold`, `truth_barrel_radius`,
  `truth_n_barrel_sides`, `truth_endcap_z`).
- Magnetic field strength (`bFieldTesla`), used for track pT reconstruction
  from curvature.
- `reassign_low_p_muons` / `muon_to_charged_hadron_p_threshold`: reassigns a
  charged candidate predicted as muon below a momentum threshold to charged
  hadron instead (on by default) -- a punch-through/misidentified hadron is
  more likely than a genuine muon at that momentum.
- `writeUnassociatedTracks`: write tracks never assigned to any shower into
  the `HitPFUnassociatedTracks` output collection (off by default) -- e.g.
  for invariant-mass calculations that want to recover tracks `HitPF` itself
  drops (charged particles that curl up before reaching the calorimeter).


## HitPF output collection conventions


- **Reference point** (`referencePoint`): for charged particles this is the
  energy-weighted shower barycenter *minus* the driving track's
  calorimeter-entry position (`PFParticleBuilder.cpp::buildChargedRecoInfo`),
  matching Python's `PickPAtDCA.predict()` (`tools_for_regression.py`:
  `barycenters - p_xyz`). For neutral
  particles, `referencePoint` is instead an absolute shower-barycenter position
  (`computeNeutralReferencePoint`)

- **Energy** (`energy`): for neutral particles this is a regression output.
  For charged particles it is the driving track's momentum magnitude `|p|`.

- **Predicted class (`HitPFIDs.type`)**: the canonical field for "which of
  the 5 classes did the model predict" is `HitPFIDs`' `type`
  (`ParticleID.type`, set via `pid.setType(recoInfo.physicsClass)`) --
  `0`=electron, `1`=charged hadron, `2`=neutral hadron, `3`=photon,
  `4`=muon. Use **this**, not PDG, to identify the predicted class: PDG
  (both `HitPF.PDG` and `HitPFIDs.PDG`) is *derived from* `type` via a fixed
  mass/species assumption per class (see below), not an independent
  measurement -- e.g. every charged hadron gets PDG ±211 (pion) regardless
  of whether it's actually a pion, kaon, or proton/

- **Mass / PDG**: fixed per predicted class, not species-specific --
  electron 0.511 MeV / PDG ±11, charged hadron **assumed pion** 139.57 MeV /
  PDG ±211, neutral hadron **assumed neutron** 939.57 MeV / PDG 2112, photon
  0 / PDG 22, muon 105.658 MeV / PDG ±13
  (`PFParticleBuilder.cpp::massFromPredictedClass`,
  `pdgFromChargedClass`/`pdgFromNeutralClass`). Note that the PID classifier only differentiates e, gamma, CH, NH, mu.

- **Charge**: derived from the driving track's curvature sign
  (`chargeSignFromTrack`, `omega > 0` -> +1), 0 for neutral particles.

- **Tracks**: only the single driving track (lowest chi2/ndf,
  `pickBestTrack`) is attached to a charged `HitPF` particle's `tracks`
  relation, even if DPC clustered more than one track into the same shower.
  Recover unassociated tracks via the
  `HitPFUnassociatedTracks` subset collection (e.g. for invariant-mass
  calculations that want to include tracks `HitPF` itself drops).

- **`HitPFIDs.parameters`**: the full per-class softmax vector (not just the
  winning class), in `chargedClassMap`/`neutralClassMap` order
  (`PFHitML.cpp`) -- 3 entries `{e, CH, mu}` for charged particles, 2 entries
  `{NH, gamma}` for neutral. Lets a downstream consumer define their own PID
  working point instead of only seeing the argmax decision already baked
  into `physicsClass`/`mass`/`PDG`.

## References

1. [arXiv:2603.04084](https://arxiv.org/html/2603.04084v1) -- hit-based
   particle-flow algorithm this package implements.
2. [HitPF](https://github.com/doloresgarcia/HitPF) -- Python training/inference
   reference implementation.
3. [pgoltstein/densitypeakclustering](https://github.com/pgoltstein/densitypeakclustering)
   -- reference for the Density Peak Clustering algorithm reimplemented in
   `Clustering.h/.cpp`.

---

*Parts of this codebase and README were drafted with AI assistance, then
reviewed and adapted if necessary.*

