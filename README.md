# k4PFHitML

`k4PFHitML` is the key4hep/Gaudi inference implementation of the hit-based
particle-flow algorithm described in
[arXiv:2603.04084](https://arxiv.org/html/2603.04084v1). It reads EDM4hep
calorimeter hits and tracks, runs the trained models (ONNX), and writes
particle-flow objects as standard EDM4hep collections. The Python reference
implementation used for training lives at
[github.com/doloresgarcia/HitPF](https://github.com/doloresgarcia/HitPF).

Originally developed for **CLD**; generalisation to other detectors is ongoing
(an `ILD` preset exists). Detector-specific inputs are the collection names,
the barrel/endcap geometry constants and the trained models.

## Quick start

From the root of the repository:

``` bash
source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh

mkdir -p models
for model in clustering_paper.onnx \
             energy_correction_paper_neutral.onnx \
             energy_correction_paper_charged_pid.onnx; do
  curl -L -o models/${model} "https://key4hep.web.cern.ch/testFiles/k4PFHitML/${model}"
done

curl -LO "https://cernbox.cern.ch/remote.php/dav/public-files/162pSTet8hUf4xD/out_reco_edm4hep_REC.edm4hep.root"

k4run k4PF-HitML/options/PerformMLPF.py \
    --inputFiles out_reco_edm4hep_REC.edm4hep.root \
    --outputFile output_HitPF.edm4hep.root \
    -n 3
```

The models (about 120 MB) are published at
<https://key4hep.web.cern.ch/testFiles/k4PFHitML/>; `models/` is where
`PerformMLPF.py` looks by default. The code
requires a key4hep nightly (EDM4hep newer than 1.0, for `Track::getTrackState`);
tested with the nightly of 2026-11-13. The example file is Z→ss. Run
`PerformMLPF.py --help` for all options.

## Inputs

A **reconstructed** EDM4hep file, e.g. the `*_REC.edm4hep.root` output of
`CLDReconstruction.py` from [CLDConfig](https://github.com/key4hep/CLDConfig).
Raw `ddsim` output is not enough: the collections below exist only after
digitisation and reconstruction.

- calorimeter hits (CLD default: `ECALBarrel`, `ECALEndcap`, `HCALBarrel`,
  `HCALEndcap`, `HCALOther`, `MUON`),
- tracks (CLD default: `SiTracks_Refitted`),
- for truth linking only: `CalohitMCTruthLink`, `SiTracksMCTruthLink`.

## Outputs

The output file contains the input collections (`keep *`) plus:

| Collection | Type | Content |
|---|---|---|
| `HitPF` | `ReconstructedParticleCollection` | Particle-flow objects: momentum, energy, mass, charge, PDG, reference point |
| `HitPFIDs` | `ParticleIDCollection` | PID per particle: type, likelihood, PDG, full per-class score vector |
| `HitPFClusters` | `ClusterCollection` | One cluster per `HitPF` particle with calorimeter hits (none for track-only charged), linked via `HitPF.clusters`: hits, energy (regressed for neutrals, hit sum for charged), energy-weighted position |
| `HitPFUnassociatedTracks` | `TrackCollection` (optional) | Tracks in no shower (DPC noise, or removed by the E/p cut), so on no `HitPF` particle |
| `HitPFMCTruthLink` | `RecoMCParticleLinkCollection` | Reco-to-truth links, weight = IoU (see [Truth linking](#truth-linking)) |

### Conventions

| Property | Charged | Neutral |
|---|---|---|
| Energy | Driving track's `\|p\|` | Regression output |
| Charge | Sign of the driving track's curvature (`omega > 0` -> +1) | 0 |
| Reference point | Driving track's point of closest approach, `(D0·cos φ, D0·sin φ, Z0)` from its `AtIP` state | Energy-weighted barycentre of all shower hits (the direction uses the ECAL-only barycentre) |
| Tracks | Only the driving track (lowest χ²/ndf, `pickBestTrack`), even if DPC clustered several into the shower | None |
| `HitPFIDs.parameters` | Softmax vector `{e, CH, mu}` | Softmax vector `{NH, gamma}` |

The predicted class is `HitPFIDs.type`. PDG and mass are fixed per class, not
measured:

| `type` | Class | PDG | Mass |
|---|---|---|---|
| 0 | Electron | ±11 | 0.511 MeV |
| 1 | Charged hadron (assumed pion) | ±211 | 139.57 MeV |
| 2 | Neutral hadron (assumed neutron) | 2112 | 939.57 MeV |
| 3 | Photon | 22 | 0 |
| 4 | Muon | ±13 | 105.658 MeV |

Tracks in no shower are in `HitPFUnassociatedTracks`, e.g. to add them back
for invariant masses.

### Reading the output

Invariant mass of all `HitPF` particles per event (about 91 GeV for the Z→ss
example):

``` python
import math
from podio.root_io import Reader

reader = Reader("output_HitPF.edm4hep.root")
for event in reader.get("events"):
    energy, px, py, pz = 0.0, 0.0, 0.0, 0.0
    for particle in event.get("HitPF"):
        momentum = particle.getMomentum()
        energy += particle.getEnergy()
        px += momentum.x
        py += momentum.y
        pz += momentum.z
    mass = math.sqrt(max(energy**2 - px**2 - py**2 - pz**2, 0))
    print(f"{len(event.get('HitPF'))} particles, invariant mass {mass:.1f} GeV")
```

## Configuration

Defaults are those of `options/PerformMLPF.py`. `--detector` (`CLD`, default,
or `ILD`) sets the input collection names, compact file and truth-matching
geometry for both algorithms.

`PFHitML`:

| Command line | Default | Description |
|---|---|---|
| `--onnx_model_clustering` | `models/clustering_paper.onnx` | Clustering model |
| `--onnx_model_properties_neutral` | `models/energy_correction_paper_neutral.onnx` | Neutral energy regression and PID |
| `--onnx_model_properties_charged` | `models/energy_correction_paper_charged_pid.onnx` | Charged PID |
| `--dpc_d_c` | 0.1 | DPC kernel width |
| `--dpc_rho_min` | 0.05 | DPC minimum density |
| `--dpc_delta_min` | 0.4 | DPC minimum distance to higher density |
| `--dpc_core_radius` | 0.5 | DPC core radius |
| `--compactFile` | from `--detector` | Detector geometry; B<sub>z</sub> at the origin is used for track p<sub>T</sub> |
| `--reassign_low_p_muons`, `--no_reassign_low_p_muons` | on | Reassign predicted muons below the threshold to charged hadrons (punch-through is more likely) |
| `--muon_to_charged_hadron_p_threshold` | 1.0 GeV | Threshold for the above |
| `--write_unassociated_tracks`, `--no_write_unassociated_tracks` | on | Write `HitPFUnassociatedTracks` |
| `--no_onnx_arena_shrink` | arena shrink on | Keep ONNX Runtime's CPU arena at its high-water mark (up to ~2 GB on CLD Z events) instead of releasing it after each inference call; shrinking saves 1-2 GB RSS for ~1% CPU |
| `--onnx_intra_op_threads` | 1 | ONNX Runtime threads within one inference call |

`HitPFRecoMCTruthLinker`:

| Command line | Default | Description |
|---|---|---|
| `--truth_iou_threshold` | 0.25 | Minimum IoU for a match |
| `--truth_barrel_radius` | from `--detector` (CLD: 2150 mm) | Calorimeter barrel inner radius, for the in-calorimeter vertex check |
| `--truth_n_barrel_sides` | from `--detector` (CLD: 12) | Barrel polygon sides |
| `--truth_endcap_z` | from `--detector` (CLD: 2307 mm) | Calorimeter endcap \|z\| |

## Truth linking

`HitPFRecoMCTruthLinker` runs after `PFHitML`. It matches reconstructed
particles to MCParticles by IoU (intersection-over-union) of their
calorimeter hits and tracks, resolved with Hungarian matching
(`TruthMatcher.h`). It works on any `ReconstructedParticleCollection`, e.g.
`PandoraPFOs`. Since a `HitPF` particle carries one track, only that track
enters the matching.

Use it for fake rate and efficiency: a reconstructed particle without a match
is a fake; an MCParticle without a match is missed.

## How it works

`PFHitML` runs two model stages per event:

1. **Clustering model** -- embeds every calorimeter hit and track and predicts
   a condensation (beta) value per node. Density Peak Clustering (DPC) on the
   embedding groups the nodes of one particle into a `Shower`. DPC is a
   reimplementation of
   [pgoltstein/densitypeakclustering](https://github.com/pgoltstein/densitypeakclustering).
2. **Regression/PID model** -- per shower, with separate charged and neutral
   models: regresses the energy and classifies electron / charged hadron /
   neutral hadron / photon / muon.

Both models are trained on CLD full simulation: clustering on Z→uds,
regression on a jet-like particle gun.

## Source files

In `k4PF-HitML/src/components/`:

| File | Content |
|---|---|
| `PFHitML.cpp` | Main Gaudi `MultiTransformer`: data prep -> clustering ONNX -> DPC -> shower building -> regression ONNX -> particle assembly |
| `HitPFRecoMCTruthLinker.cpp` | Gaudi `Transformer` producing the reco-MC truth links |
| `DataPreprocessing.h/.cpp` | Clustering model input graph; per-shower regression inputs |
| `Clustering.h/.cpp` | DPC, plus the E/p consistency cut (`remove_bad_tracks_from_cluster`) that can evict a track |
| `ShowerBuilder.h/.cpp` | DPC cluster labels -> `Shower` objects |
| `Shower.h/.cpp` | Per-shower container (hits, tracks, betas) and derived quantities |
| `PFParticleBuilder.h/.cpp` | `Shower` + regression/PID output -> `HitPF`/`HitPFIDs` entry, with the conventions above |
| `TruthMatcher.h` | IoU-based Hungarian matching |
| `EvaluationSummary.h/.cpp` | Per-event reco/truth summary rows (optional; validation/debug logging) |
| `ONNXHelper.h/.cpp` | Thin ONNX Runtime wrapper |
| `Helpers.h/.cpp` | Shared small utilities |

## References

1. [arXiv:2603.04084](https://arxiv.org/html/2603.04084v1) -- the algorithm.
2. [HitPF](https://github.com/doloresgarcia/HitPF) -- Python training/inference
   reference implementation.
3. [pgoltstein/densitypeakclustering](https://github.com/pgoltstein/densitypeakclustering)
   -- reference DPC implementation.

---

*Parts of this codebase and README were drafted with AI assistance, then
reviewed and adapted if necessary.*
