# coding: utf-8

"""
Configuration of the hhh4b2w analysis.
"""

import law
import order as od
from scinum import Number

from columnflow.config_util import (
    get_root_processes_from_campaign, add_shift_aliases, add_category, verify_config_processes,
)
from columnflow.columnar_util import EMPTY_FLOAT, ColumnCollection, skip_column
from columnflow.util import DotDict, maybe_import
from hhh4b2w.config.categories import add_categories
ak = maybe_import("awkward")


#
# the main analysis object
#

analysis_hhh4b2w = ana = od.Analysis(
    name="analysis_hhh4b2w",
    id=1,
)

# analysis-global versions
# (see cfg.x.versions below for more info)
ana.x.versions = {}

# files of bash sandboxes that might be required by remote tasks
# (used in cf.HTCondorWorkflow)
ana.x.bash_sandboxes = ["$CF_BASE/sandboxes/cf.sh", "$HHH4B2W_BASE/sandboxes/venv_ml.sh"]
default_sandbox = law.Sandbox.new(law.config.get("analysis", "default_columnar_sandbox"))
if default_sandbox.sandbox_type == "bash" and default_sandbox.name not in ana.x.bash_sandboxes:
    ana.x.bash_sandboxes.append(default_sandbox.name)

# files of cmssw sandboxes that might be required by remote tasks
# (used in cf.HTCondorWorkflow)
ana.x.cmssw_sandboxes = [
    # "$CF_BASE/sandboxes/cmssw_default.sh",
]

# config groups for conveniently looping over certain configs
# (used in wrapper_factory)
ana.x.config_groups = {}

# named function hooks that can modify store_parts of task outputs if needed
ana.x.store_parts_modifiers = {}


#
# setup configs
#

# an example config is setup below, based on cms NanoAOD v9 for Run2 2017, focussing on
# ttbar and single top MCs, plus single muon data
# update this config or add additional ones to accomodate the needs of your analysis

from cmsdb.campaigns.run3_2022_preEE_nano_v12 import campaign_run3_2022_preEE_nano_v12

# copy the campaign
# (creates copies of all linked datasets, processes, etc. to allow for encapsulated customization)
campaign = campaign_run3_2022_preEE_nano_v12.copy()

# get all root processes
procs = get_root_processes_from_campaign(campaign)

# create a config by passing the campaign, so id and name will be identical
cfg = ana.add_config(campaign)

# gather campaign data
# year = campaign.x.year
# year = 2017
year = 2022

# add processes we are interested in
process_names = [
    "data",
    "tt",
    "w_lnu",
    "hhh",
    "tth",
    "hh_ggf",
    "ttHH"
]

for process_name in process_names:
    # add the process
    proc = cfg.add_process(procs.get(process_name))

    # configuration of colors, labels, etc. can happen here
    if proc.is_mc:
        proc.color1 = (244, 182, 66) if proc.name == "tt" else (244, 93, 66)

# add datasets we need to study
dataset_names = [
    # HHH -> 4b2W
    
    "hhh_bbbbww_c3_0_d4_0_amcatnlo",
    "hhh_bbbbww_c3_0_d4_99_amcatnlo",
    "hhh_bbbbww_c3_0_d4_m1_amcatnlo",
    "hhh_bbbbww_c3_19_d4_19_amcatnlo",
    "hhh_bbbbww_c3_1_d4_0_amcatnlo",
    "hhh_bbbbww_c3_1_d4_2_amcatnlo",
    # "hhh_bbbbww_c3_2_d4_m1_amcatnlo",
    "hhh_bbbbww_c3_4_d4_9_amcatnlo",
    "hhh_bbbbww_c3_m1_d4_0_amcatnlo",
    "hhh_bbbbww_c3_m1_d4_m1_amcatnlo",
    "hhh_bbbbww_c3_m1p5_d4_m0p5_amcatnlo",

    # data
    "data_mu_c",

    # backgrounds
    # tt
    "tt_sl_powheg",
    "tt_dl_powheg",
    "tt_fh_powheg",
    # W + Jets
    "w_lnu_amcatnlo",
    #Di-Higgs
    "hh_ggf_hbb_hvv_kl1_kt1_powheg",
    #ttH
    "tth_hbb_powheg",
    #ttHH
    "ttHHto4b_madgraph",

    ]

for dataset_name in dataset_names:
    # add the dataset
    dataset = cfg.add_dataset(campaign.get_dataset(dataset_name))

    # #for testing purposes, limit the number of files to 10
    # for info in dataset.info.values():
    #     info.n_files = min(info.n_files, 1)

    if "hhh_bbbbww" in dataset.name:
       dataset.add_tag("HHH")
    elif "hh_ggf" in dataset.name:
        dataset.add_tag("HH")
    elif "tth_hbb" in dataset.name:
        dataset.add_tag("H")
    elif "ttHHto4b" in dataset.name:
        dataset.add_tag("HH") 
    elif "tt_" in dataset.name:
        dataset.add_tag("tt")


# verify that the root process of all datasets is part of any of the registered processes
verify_config_processes(cfg, warn=True)

# default objects, such as calibrator, selector, producer, ml model, inference model, etc
cfg.x.default_calibrator = "example"
cfg.x.default_selector = "example"
cfg.x.default_producer = "example"
cfg.x.default_weight_producer = "example"
# cfg.x.default_ml_model = "example"
# cfg.x.default_ml_model = "DNN"
cfg.x.default_inference_model = "example"
cfg.x.default_categories = ("incl",)
cfg.x.default_variables = ("n_jet", "jet1_pt")
cfg.x.default_bins_per_category = {
    "incl": 10,
    }
# process groups for conveniently looping over certain processs
# (used in wrapper_factory and during plotting)
cfg.x.process_groups = {}

# dataset groups for conveniently looping over certain datasets
# (used in wrapper_factory and during plotting)
cfg.x.dataset_groups = {}

# category groups for conveniently looping over certain categories
# (used during plotting)
cfg.x.category_groups = {}

# variable groups for conveniently looping over certain variables
# (used during plotting)
cfg.x.variable_groups = {}

# shift groups for conveniently looping over certain shifts
# (used during plotting)
cfg.x.shift_groups = {}

# general_settings groups for conveniently looping over different values for the general-settings parameter
# (used during plotting)
cfg.x.general_settings_groups = {}

# process_settings groups for conveniently looping over different values for the process-settings parameter
# (used during plotting)
cfg.x.process_settings_groups = {}

# variable_settings groups for conveniently looping over different values for the variable-settings parameter
# (used during plotting)
cfg.x.variable_settings_groups = {}

# custom_style_config groups for conveniently looping over certain style configs
# (used during plotting)
cfg.x.custom_style_config_groups = {
    "small_legend": {
    "legend_cfg": {"ncols": 2, "fontsize": 14},
    },
    "example": {
        "legend_cfg": {"title": "my custom legend title", "ncols": 2},
        "ax_cfg": {"ylabel": "my ylabel", "xlim": (0, 100)},
        "rax_cfg": {"ylabel": "some other ylabel"},
        "annotate_cfg": {"text": "category label usually here"},
    },
}
cfg.x.default_custom_style_config = "small_legend"

# selector step groups for conveniently looping over certain steps
# (used in cutflow tasks)
cfg.x.selector_step_groups = {
    "default": ["muon", "jet"],
}

# calibrator groups for conveniently looping over certain calibrators
# (used during calibration)
cfg.x.calibrator_groups = {}

# producer groups for conveniently looping over certain producers
# (used during the ProduceColumns task)
cfg.x.producer_groups = {}

# ml_model groups for conveniently looping over certain ml_models
# (used during the machine learning tasks)
cfg.x.ml_model_groups = {}

# custom method and sandbox for determining dataset lfns
cfg.x.get_dataset_lfns = None
cfg.x.get_dataset_lfns_sandbox = None

# whether to validate the number of obtained LFNs in GetDatasetLFNs
# (currently set to false because the number of files per dataset is truncated to 2)
cfg.x.validate_dataset_lfns = False

# lumi values in inverse pb
# https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun2?rev=2#Combination_and_correlations
# cfg.x.luminosity = Number(41480, {
#     "lumi_13TeV_2017": 0.02j,
#     "lumi_13TeV_1718": 0.006j,
#     "lumi_13TeV_correlated": 0.009j,
# })

cfg.x.luminosity = Number(7971, {
                "lumi_13TeV_2022": 0.01j,
                #"lumi_13TeV_correlated": 0.006j,
            })

# b-tag working points
# main source with all WPs: https://btv-wiki.docs.cern.ch/ScaleFactors/#sf-campaigns
# 2016/17 sources with revision:
# https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL16preVFP?rev=6
# https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL16postVFP?rev=8
# https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL17?rev=15
# https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL17?rev=17

corr_postfix = ""
# if year == 2016:
#     corr_postfix = f"{campaign.x.vfp}VFP"
# elif year == 2022:
#     corr_postfix = f"{campaign.x.EE}EE"
corr_postfix = f"{'post' if campaign.has_tag('EE') else 'pre'}EE"

if year != 2017 and year != 2022:
    raise NotImplementedError("For now, only 2017 and 2022 campaign is implemented")

# add some important tags to the config
cfg.x.cpn_tag = f"{year}{corr_postfix}"

cfg.x.btag_working_points = DotDict.wrap({
    "deepjet": {
        "loose": {"2016preVFP": 0.0508, "2016postVFP": 0.0480, "2017": 0.0532, "2018": 0.0490, "2022preEE": 0.0583, "2022postEE": 0.0614, "2023": 0.0479, "2023BPix": 0.048}.get(cfg.x.cpn_tag, 0.0),  # noqa
        "medium": {"2016preVFP": 0.2598, "2016postVFP": 0.2489, "2017": 0.3040, "2018": 0.2783, "2022preEE": 0.3086, "2022postEE": 0.3196, "2023": 0.2431, "2023BPix": 0.2435}.get(cfg.x.cpn_tag, 0.0),  # noqa
        "tight": {"2016preVFP": 0.6502, "2016postVFP": 0.6377, "2017": 0.7476, "2018": 0.7100, "2022preEE": 0.7183, "2022postEE": 0.73, "2023": 0.6553, "2023BPix": 0.6563}.get(cfg.x.cpn_tag, 0.0),  # noqa
    },
    "deepcsv": {
        "loose": {"2016preVFP": 0.2027, "2016postVFP": 0.1918, "2017": 0.1355, "2018": 0.1208}.get(cfg.x.cpn_tag, 0.0),  # noqa
        "medium": {"2016preVFP": 0.6001, "2016postVFP": 0.5847, "2017": 0.4506, "2018": 0.4168}.get(cfg.x.cpn_tag, 0.0),  # noqa
        "tight": {"2016preVFP": 0.8819, "2016postVFP": 0.8767, "2017": 0.7738, "2018": 0.7665}.get(cfg.x.cpn_tag, 0.0),  # noqa
    },
    "particlenet": {
        "loose": {"2022preEE": 0.047, "2022postEE": 0.0499, "2023": 0.0358, "2023BPix": 0.359}.get(cfg.x.cpn_tag, 0.0),  # noqa
        "medium": {"2022preEE": 0.245, "2022postEE": 0.2605, "2023": 0.1917, "2023BPix": 0.1919}.get(cfg.x.cpn_tag, 0.0),  # noqa
        "tight": {"2022preEE": 0.6734, "2022postEE": 0.6915, "2023": 0.6172, "2023BPix": 0.6133}.get(cfg.x.cpn_tag, 0.0),  # noqa
    },
})

# names of muon correction sets and working points
# (used in the muon producer)
#cfg.x.muon_sf_names = ("NUM_TightRelIso_DEN_TightIDandIPCut", f"{year}_UL")
cfg.x.muon_sf_id_names = ("NUM_HighPtID_DEN_TrackerMuons", f"{year}{corr_postfix}")
cfg.x.muon_sf_iso_names = ("NUM_TightRelTkIso_DEN_HighPtID", f"{year}{corr_postfix}")


# register shifts
cfg.add_shift(name="nominal", id=0)

# tune shifts are covered by dedicated, varied datasets, so tag the shift as "disjoint_from_nominal"
# (this is currently used to decide whether ML evaluations are done on the full shifted dataset)
cfg.add_shift(name="tune_up", id=1, type="shape", tags={"disjoint_from_nominal"})
cfg.add_shift(name="tune_down", id=2, type="shape", tags={"disjoint_from_nominal"})

# fake jet energy correction shift, with aliases flaged as "selection_dependent", i.e. the aliases
# affect columns that might change the output of the event selection
cfg.add_shift(name="jec_up", id=20, type="shape")
cfg.add_shift(name="jec_down", id=21, type="shape")
add_shift_aliases(
    cfg,
    "jec",
    {
        "Jet.pt": "Jet.pt_{name}",
        "Jet.mass": "Jet.mass_{name}",
        "MET.pt": "MET.pt_{name}",
        "MET.phi": "MET.phi_{name}",
    },
)

# event weights due to muon scale factors
cfg.add_shift(name="mu_up", id=10, type="shape")
cfg.add_shift(name="mu_down", id=11, type="shape")
add_shift_aliases(cfg, "mu", {"muon_weight": "muon_weight_{direction}"})

# external files
json_mirror = "/afs/cern.ch/user/m/mfrahm/public/mirrors/jsonpog-integration-a332cfa"
cfg.x.external_files = DotDict.wrap({
    # lumi files
    "lumi": {
        "golden": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt", "v1"),  # noqa
        "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
    },
    # 2022 version
    # "lumi": {
    #     "golden": ("https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/Cert_Collisions2022_355100_362760_Golden.json", "v1"),  # noqa
    #     "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
    # },

    # muon scale factors
    #"muon_sf": (f"{json_mirror}/POG/MUO/{year}_UL/muon_Z.json.gz", "v1"),
    #"muon_sf": (f"{json_mirror}/POG/MUO/{corr_tag}/muon_Z.json.gz", "v1"),

    # 2022 version
    "muon_sf": (f"{json_mirror}/POG/MUO/{year}_Summer22/muon_Z.json.gz", "v1"),
})

# target file size after MergeReducedEvents in MB
cfg.x.reduced_file_size = 512.0

# columns to keep after certain steps
cfg.x.keep_columns = DotDict.wrap({
    "cf.ReduceEvents": {
        # general event info, mandatory for reading files with coffea
        # additional columns can be added as strings, similar to object info
        ColumnCollection.MANDATORY_COFFEA,
        # object info
        "Jet.{pt,eta,phi,mass,btagDeepFlavB,hadronFlavour}",
        "BJet.{pt,eta,phi,mass,btagDeepFlavB,hadronFlavour}",
        "Muon.{pt,eta,phi,mass,pfRelIso04_all,highPtID,tkIsoID}",
        "Electron.{pt,eta,phi,mass,pfRelIso04_all,mvaIso_WP80}",
        "lepton.{pt,eta,phi,mass}",
        "MET.{pt,phi,significance,covXX,covXY,covYY}",
        "PV.npvs",
        # all columns added during selection using a ColumnCollection flag, but skip cutflow ones
        ColumnCollection.ALL_FROM_SELECTOR,
        skip_column("cutflow.*"),
        "gen_hhh4b2w_decay","GenPart.*"
        #"gen_hhh4b2w_decay_b","GenPart.*"
    },
    "cf.MergeSelectionMasks": {
        "cutflow.*",
    },
    "cf.UniteColumns": {
        "*",
    },
})

# # pinned versions
# # (see [versions] in law.cfg for more info)
# prod_version = "HHH_v5"
# prod_version2 = "HHH_v6"
# cfg.x.versions = {
#     "cf.CalibrateEvents": prod_version,
#     "cf.SelectEvents": prod_version,
#     "cf.MergeSelectionStats": prod_version,
#     "cf.MergeSelectionMasks": prod_version,
#     "cf.ReduceEvents": prod_version,
#     "cf.MergeReductionStats": prod_version,
#     "cf.MergeReduceEvents": prod_version,
#     "cf.ProvideReducedEvents": prod_version,
#     "cf.ProduceColumns": prod_version2,
#     "cf.MergeMLEvents": prod_version2,
#     "cf.MergeMLStats": prod_version2,
#     "cf.PrepareMLEvents": prod_version2,
#     "cf.MLTraining": prod_version2,
# }

cfg.x.versions = {}


# channels
# (just one for now)
cfg.add_channel(name="mutau", id=1)

# histogramming hooks, invoked before creating plots when --hist-hook parameter set
cfg.x.hist_hooks = {}

is_signal_sm = lambda proc_name: "c3_0_d4_0" in proc_name
# is_signal_sm_ggf = lambda proc_name: "kl1_kt1" in proc_name
# is_signal_sm_vbf = lambda proc_name: "kv1_k2v1_kl1" in proc_name
# is_gghh_sm = lambda proc_name: "kl1_kt1" in proc_name
# is_qqhh_sm = lambda proc_name: "kv1_k2v1_kl1" in proc_name
# is_signal_ggf_kl1 = lambda proc_name: "kl1_kt1" in proc_name and "hh_ggf" in proc_name
# is_signal_vbf_kl1 = lambda proc_name: "kv1_k2v1_kl1" in proc_name and "hh_vbf" in proc_name
is_background = lambda proc_name: ("hhh_bbbbww" not in proc_name)

cfg.x.inference_category_rebin_processes = {
        "incl": is_signal_sm
    }
# add categories using the "add_category" tool which adds auto-generated ids
# the "selection" entries refer to names of categorizers, e.g. in categorization/example.py
# note: it is recommended to always add an inclusive category with id=1 or name="incl" which is used
#       in various places, e.g. for the inclusive cutflow plots and the "empty" selector

add_categories(cfg)

# add_category(
#     cfg,
#     id=1,
#     name="incl",
#     selection="cat_incl",
#     label="inclusive",
# )
# add_category(
#     cfg,
#     id=2,
#     name="2j",
#     selection="cat_2j",
#     label="2 jets",
# )


# add variables
# (the "event", "run" and "lumi" variables are required for some cutflow plotting task,
# and also correspond to the minimal set of columns that coffea's nano scheme requires)
cfg.add_variable(
    name="event",
    expression="event",
    binning=(1, 0.0, 1.0e9),
    x_title="Event number",
    discrete_x=True,
)
cfg.add_variable(
    name="run",
    expression="run",
    binning=(1, 100000.0, 500000.0),
    x_title="Run number",
    discrete_x=True,
)
cfg.add_variable(
    name="lumi",
    expression="luminosityBlock",
    binning=(1, 0.0, 5000.0),
    x_title="Luminosity block",
    discrete_x=True,
)
cfg.add_variable(
    name="n_jets",
    expression="n_jets",
    binning=(11, -0.5, 10.5),
    x_title="Number of jets",
    discrete_x=True,
)
# pt of all jets in every event
cfg.add_variable(
    name="jets_pt",
    expression="Jet.pt",
    binning=(40, 0.0, 400.0),
    unit="GeV",
    x_title=r"$p_{{T}}$ of all jets",
)
# # pt of the first jet in every event
# cfg.add_variable(
#     name="jet1_pt",  # variable name, to be given to the "--variables" argument for the plotting task
#     expression="Jet.pt[:,0]",  # content of the variable
#     null_value=EMPTY_FLOAT,  # value to be given if content not available for event
#     binning=(40, 0.0, 400.0),  # (bins, lower edge, upper edge)
#     unit="GeV",  # unit of the variable, if any
#     x_title=r"Jet 1 $p_{T}$",  # x title of histogram when plotted
# )
# # eta of the first jet in every event
# cfg.add_variable(
#     name="jet1_eta",
#     expression="Jet.eta[:,0]",
#     null_value=EMPTY_FLOAT,
#     binning=(30, -3.0, 3.0),
#     x_title=r"Jet 1 $\eta$",
# )

cfg.add_variable(
    name="ht",
    expression=lambda events: ak.sum(events.Jet.pt, axis=1),
    binning=(40, 0.0, 800.0),
    unit="GeV",
    x_title="HT",
)
# weights
cfg.add_variable(
    name="mc_weight",
    expression="mc_weight",
    binning=(200, -10, 10),
    x_title="MC weight",
)
# cutflow variables
cfg.add_variable(
    name="cf_jet1_pt",
    expression="cutflow.jet1_pt",
    binning=(40, 0.0, 400.0),
    unit="GeV",
    x_title=r"Jet 1 $p_{T}$",
)


# Muon Pt

cfg.add_variable(
    name="muon_pt",
    expression="Muon.pt",
    null_value=EMPTY_FLOAT,
    binning=(40, 0.0, 400.0),
    unit="GeV",
    x_title=r"Muon $p_{{T}}$",
)

cfg.add_variable(
    name="muon_eta",
    expression="Muon.eta",
    null_value=EMPTY_FLOAT,
    binning=(30, -3.0, 3.0),
    unit="GeV",
    x_title=r"Muon $\eta$",
)

cfg.add_variable(
    name="n_muons",
    expression="n_muon",
    null_value=EMPTY_FLOAT,
    binning=(4, -0.5, 3.5),
    unit="",
    x_title=r"Number of muons",
)

cfg.add_variable(
    name="electron_pt",
    expression="Electron.pt",
    null_value=EMPTY_FLOAT,
    binning=(40, 0.0, 400.0),
    unit="GeV",
    x_title=r"Electron $p_{{T}}$",
)

cfg.add_variable(
    name="electron_eta",
    expression="Electron.eta",
    null_value=EMPTY_FLOAT,
    binning=(30, -3.0, 3.0),
    unit="GeV",
    x_title=r"Electron $\eta",
)
cfg.add_variable(
    name="n_electrons",
    expression="n_electron",
    null_value=EMPTY_FLOAT,
    binning=(4, -0.5, 3.5),
    unit="",
    x_title=r"Number of electrons",
)
cfg.add_variable(
    name="n_lepton",
    expression="n_lepton",
    null_value=EMPTY_FLOAT,
    binning=(4, -0.5, 3.5),
    unit="",
    x_title=r"Number of leptons",
)

# Jet Pt & Eta
for i in range(0, 7, 1):

    cfg.add_variable(
        name=f"jet{i+1}_pt",  # variable name, to be given to the "--variables" argument for the plotting task
        expression=f"Jet.pt[:,{i}]",  # content of the variable
        null_value=EMPTY_FLOAT,  # value to be given if content not available for event
        binning=(40, 0.0, 400.0),  # (bins, lower edge, upper edge)
        unit="GeV",  # unit of the variable, if any
        x_title=rf"Jet {i+1} $p_{{T}}$",  # x title of histogram when plotted
    )

    cfg.add_variable(
        name=f"jet{i+1}_eta",
        expression=f"Jet.eta[:,{i}]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=rf"Jet {i+1} $\eta$",
    )

# B Jet Pt & Eta
for i in range(0, 7, 1):

    cfg.add_variable(
        name=f"BJet{i+1}_pt",  # variable name, to be given to the "--variables" argument for the plotting task
        expression=f"BJet.pt[:,{i}]",  # content of the variable
        null_value=EMPTY_FLOAT,  # value to be given if content not available for event
        binning=(40, 0.0, 400.0),  # (bins, lower edge, upper edge)
        unit="GeV",  # unit of the variable, if any
        x_title=rf"B-Jet {i+1} $p_{{T}}$",  # x title of histogram when plotted
    )

    cfg.add_variable(
        name=f"BJet{i+1}_eta",
        expression=f"BJet.eta[:,{i}]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=rf"B-Jet {i+1} $\eta$",
    )

cfg.add_variable(
    name="n_bjet",
    expression="n_bjet",
    binning=(11, -0.5, 10.5),
    x_title="Number of B-Jets",
)

# Generator Level Variables

# Higgs
cfg.add_variable(
    name="gen_hhh_mass",
    expression="gen_hhh4b2w_decay.hhh.mass",
    binning=(40, 300., 1000.),
    unit="GeV",
    x_title=r"$m_{HHH}^{gen}$",
)

for i in range(1, 4, 1):
    cfg.add_variable(
        name=f"gen_h{i}_pt",
        expression=f"gen_hhh4b2w_decay.h{i}.pt",
        binning=(40, 0., 400.),
        unit="GeV",
        x_title=rf"$p_{{T, H_{i}}}^{{gen}}$",
    )

# b-quarks
for i in range(1, 5, 1):
    cfg.add_variable(
        name=f"gen_b{i}_pt",
        expression="gen_hhh4b2w_decay.h1.pt",
        binning=(40, 0., 400.),
        unit="GeV",
        x_title=rf"$p_{{T, b_{i}}}^{{gen}}$",
    )

# W-bosons
cfg.add_variable(
    name="gen_Wlep_pt",
    expression="gen_hhh4b2w_decay.wlep.pt",
    binning=(40, 0., 400.),
    unit="GeV",
    x_title=r"$p_{T, W_{lep}}^{gen}$",
)

cfg.add_variable(
    name="gen_Whad_pt",
    expression="gen_hhh4b2w_decay.whad.pt",
    binning=(40, 0., 400.),
    unit="GeV",
    x_title=r"$p_{T, W_{had}}^{gen}$",
)

# Leptons
cfg.add_variable(
    name="gen_lepton_pt",
    expression="gen_hhh4b2w_decay.l.pt",
    binning=(40, 0., 400.),
    unit="GeV",
    x_title=r"$p_{T, Lepton}^{gen}$",
)

# Quarks
for i in range(1, 3, 1):
    cfg.add_variable(
        name=f"gen_q{i}_pt",
        expression=f"gen_hhh4b2w_decay.q{i}.pt",
        binning=(40, 0., 400.),
        unit="GeV",
        x_title=rf"$p_{{T, q_{i}}}^{{gen}}$",
    )


# Sensitive Variables
cfg.add_variable(
    name="m_jj",
    binning=(40, 0., 400.),
    unit="GeV",
    x_title=r"$m_{j_{1}j_{2}}$",
    )
cfg.add_variable(
    name="deltaR_jj",
    binning=(40, 0, 5),
    x_title=r"$\Delta R(j_{1},j_{2})$",
    )
cfg.add_variable(
    name="jj_pt",
    binning=(40,0.,400.),
    unit="GeV",
    x_title=r"$p_T^{j_{1}j_{2}}$"
    )
cfg.add_variable(
    name="dr_min_jj",
    binning=(40, 0, 5),
    x_title=r"$\Delta R_{min}^{(j_{1}j_{2})}$",
    )
cfg.add_variable(
    name="dr_mean_jj",
    binning=(40, 0, 5),
    x_title=r"$\Delta R_{mean}^{(j_{1}j_{2})}$",
    )


cfg.add_variable(
    name="m_bb",
    binning=(40, 0., 400.),
    unit="GeV",
    x_title=r"$m_{b_{1}b_{2}}$",
    )
cfg.add_variable(
    name="bb_pt",
    binning=(40, 0., 400.),
    unit="GeV",
    x_title=r"$p_T^{b_{1}b_{2}}$"
    )
cfg.add_variable(
    name="deltaR_bb",
    binning=(40, 0, 5),
    x_title=r"$\Delta R(b_{1} b_{2})$",
    )
cfg.add_variable(
    name="dr_min_bb",
    binning=(40, 0, 5),
    x_title=r"$\Delta R_{min}^{(b_{1} b_{2})}$",
    )
cfg.add_variable(
    name="dr_mean_bb",
    binning=(40, 0., 5),
    x_title=r"$\Delta R_{mean}^{(b_{1} b_{2})}$",
)

cfg.add_variable(
    name="deltaR_lbb",
    binning=(40, 0, 5),
    x_title=r"$\Delta R(l,b_{1} b_{2})$",
    )

cfg.add_variable(
    name="matched_jet_pt",
    expression="Bjet_matches.pt",
    binning=(40, 0., 400.),
    unit="GeV",
    x_title=r"$Jet\ matched\ p_{T}$",
)

cfg.add_variable(
    name="nonmatched_jet_pt",
    expression="Bjet_nonmatches.pt",
    binning=(40, 0., 400.),
    unit="GeV",
    x_title=r"$Jet\ non-matched\ p_{T}$",
)

cfg.add_variable(
    name="lepton_pt",
    expression="lepton.pt",
    binning=(40, 0., 400.),
    unit="GeV",
    x_title=r"$Lepton\ p_{T}$",
)
cfg.add_variable(
    name="lepton_eta",
    expression="lepton.eta",
    null_value=EMPTY_FLOAT,
    binning=(30, -3.0, 3.0),
    unit="GeV",
    x_title=rf"Lepton $\eta$",
)
cfg.add_variable(
    name="lepton_phi",
    expression="lepton.phi",
    null_value=EMPTY_FLOAT,
    binning=(30, -3.0, 3.0),
    unit="GeV",
    x_title=rf"Lepton $\phi$",
)


cfg.add_variable(
   name="DNN_v10.output",
   null_value=-1,
   binning=(15, 0, 1.0),
   x_title=f"DNN output",
)