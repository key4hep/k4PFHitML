#
# Copyright (c) 2020-2024 Key4hep-Project.
#
# This file is part of Key4hep.
# See https://key4hep.github.io/key4hep-doc/ for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from Configurables import PFHitML
#from Configurables import k4DataSvc
from Gaudi.Configuration import INFO
from k4FWCore import ApplicationMgr, IOSvc
from k4FWCore.parseArgs import parser



# parse the custom arguments
parser_group = parser.add_argument_group("PerformMLPF.py custom options")
parser_group.add_argument("--inputFiles", nargs="+", metavar=("file1", "file2"), required=True,
                          help="One or multiple input files")
parser_group.add_argument("--outputFile", default="output_HitPF.edm4hep.root")
parser_group.add_argument("--num_ev", type=int, help="Number of events to process (-1 means all)", default=-1)
parser_group.add_argument("--onnx_model_clustering", help="Path to ONNX model used for clustering", default="models/clustering_paper.onnx")
parser_group.add_argument("--onnx_model_properties_neutral", help="Path to neutral ONNX model used for energy regression and PID", default="models/energy_correction_paper_neutral.onnx")
parser_group.add_argument("--onnx_model_properties_charged", help="Path to charged ONNX model used for PID", default="models/energy_correction_paper_charged_pid.onnx")

parser_group.add_argument("--dpc_d_c", type=float, default=0.1, help="DPC clustering: Gaussian kernel bandwidth for local density")
parser_group.add_argument("--dpc_rho_min", type=float, default=0.05, help="DPC clustering: minimum local density for a point to be a cluster center")
parser_group.add_argument("--dpc_delta_min", type=float, default=0.4, help="DPC clustering: minimum distance-to-higher-density for a point to be a cluster center")
parser_group.add_argument("--dpc_core_radius", type=float, default=0.5, help="DPC clustering: max distance to a center for a hit to be kept as a core (non-halo) member")

parser_group.add_argument("--truth_iou_threshold", type=float, default=0.25, help="Truth matching: minimum IoU for a shower-to-MCParticle match")
parser_group.add_argument("--truth_barrel_radius", type=float, default=None, help="Truth matching: detector barrel radius [mm]")
parser_group.add_argument("--truth_n_barrel_sides", type=int, default=None, help="Truth matching: number of barrel polygon sides")
parser_group.add_argument("--truth_endcap_z", type=float, default=None, help="Truth matching: detector endcap |z| [mm]")

parser_group.add_argument("--bFieldTesla", type=float, default=None, help="Magnetic field strength [T] used for track pT reconstruction from track curvature")

parser_group.add_argument("--reassign_low_p_muons", dest="reassign_low_p_muons", action="store_true", default=True,
                          help="Reassign charged candidates predicted as muon with momentum below "
                               "--muon_to_charged_hadron_p_threshold to charged hadron (default: on)")
parser_group.add_argument("--no_reassign_low_p_muons", dest="reassign_low_p_muons", action="store_false",
                          help="Disable --reassign_low_p_muons")
parser_group.add_argument("--muon_to_charged_hadron_p_threshold", type=float, default=1.0,
                          help="Momentum threshold [GeV] below which a predicted muon is reassigned "
                               "to charged hadron (if --reassign_low_p_muons is set)")

parser_group.add_argument("--write_unassociated_tracks", dest="writeUnassociatedTracks", action="store_true",
                          default=False,
                          help="Write tracks not assigned to any shower into the HitPFUnassociatedTracks "
                               "output collection (default: off)")
parser_group.add_argument("--no_write_unassociated_tracks", dest="writeUnassociatedTracks", action="store_false",
                          help="Disable --write_unassociated_tracks")
parser_group.add_argument("--detector", choices=["CLD", "ILD"], default="CLD",
                          help="Detector preset: input collection names and truth-matching geometry")
args = parser.parse_known_args()[0]

DETECTOR_PRESETS = {
    "CLD": {
        "collections": {
            "EcalBarrelHits":      ["ECALBarrel"],
            "EcalEndcaplHits":     ["ECALEndcap"],      # sic: "Endcapl" is the property name in PFHitML.cpp
            "HcalBarrelHits":      ["HCALBarrel"],
            "HcalEndcapHits":      ["HCALEndcap"],
            "HcalOtherHits":       ["HCALOther"],
            "MUON":                ["MUON"],
            "Tracks":              ["SiTracks_Refitted"],
            "MCParticles":         ["MCParticles"],
            "CalohitMCTruthLink":  ["CalohitMCTruthLink"],
            "SiTracksMCTruthLink": ["SiTracksMCTruthLink"],
        },
        "truth_barrel_radius": 2150.,   # ECalBarrel_inner_radius
        "truth_n_barrel_sides": 12,
        "truth_endcap_z": 2307.,        # ECalEndcap_min_z
        "bFieldTesla": 2.0,
    },
    "ILD": {
        "collections": {
            "EcalBarrelHits":      ["EcalBarrelCollectionRec"],
            "EcalEndcaplHits":     ["EcalEndcapsCollectionRec"],
            "HcalBarrelHits":      ["HcalBarrelCollectionRec"],
            "HcalEndcapHits":      ["HcalEndcapsCollectionRec"],
            "HcalOtherHits":       ["HcalEndcapRingCollectionRec"],
            "MUON":                ["MUON"],
            "Tracks":              ["MarlinTrkTracks"],
            "MCParticles":         ["MCParticles"],
            "CalohitMCTruthLink":  ["CalohitMCTruthLink"],
            "SiTracksMCTruthLink": ["MarlinTrkTracksMCTruthLink"],
        },
        "truth_barrel_radius": 1805.,   # TPC_outer_radius 1769.8 + Ecal_Tpc_gap 35
        "truth_n_barrel_sides": 8,      # Ecal_Hcal_symmetry
        "truth_endcap_z": 2412.,        # TPC_Ecal_Hcal_barrel_halfZ 2350 + 61.8
        "bFieldTesla": 2.0,             # ILD_FCCee: 2 T (checked with track pT vs MC)
    },
}
preset = DETECTOR_PRESETS[args.detector]

def pick(name):
    value = getattr(args, name)
    return preset[name] if value is None else value

svc = IOSvc("IOSvc")
svc.Input = args.inputFiles
svc.Output = args.outputFile


Multitransformer = PFHitML("PFHitML",
                            model_path_clustering=args.onnx_model_clustering,
                            model_path_properties_neutral=args.onnx_model_properties_neutral,
                            model_path_properties_charged=args.onnx_model_properties_charged,
                            dpc_d_c=args.dpc_d_c,
                            dpc_rho_min=args.dpc_rho_min,
                            dpc_delta_min=args.dpc_delta_min,
                            dpc_core_radius=args.dpc_core_radius,
                            truth_iou_threshold=args.truth_iou_threshold,
                            truth_barrel_radius=pick("truth_barrel_radius"),
                            truth_n_barrel_sides=pick("truth_n_barrel_sides"),
                            truth_endcap_z=pick("truth_endcap_z"),
                            bFieldTesla=pick("bFieldTesla"),
                            **preset["collections"],
                            reassign_low_p_muons=args.reassign_low_p_muons,
                            muon_to_charged_hadron_p_threshold=args.muon_to_charged_hadron_p_threshold,
                            writeUnassociatedTracks=args.writeUnassociatedTracks,
                    )


svc.outputCommands = ["keep *"]

ApplicationMgr(TopAlg=[Multitransformer],
               EvtSel="NONE",
               EvtMax=args.num_ev,
               #ExtSvc=[k4DataSvc("EventDataSvc")],
               ExtSvc=[svc],
               OutputLevel=INFO,
               )

