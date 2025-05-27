# coding: utf-8

"""
Producers for ML inputs
"""
import functools
import itertools

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.production.normalization import normalization_weights
from hhh4b2w.production.sensitive_variables import jj_features, bb_features, l_bb_features


ak = maybe_import("awkward")
np = maybe_import("numpy")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")

# use float32 type for ML input columns
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses={
        # AK4 jets
        "Jet.{pt,eta,phi,mass}",
        "BJet.{pt,eta,phi,mass}",
        "Muon.{pt,eta,phi,mass}",
        "Electron.{pt,eta,phi,mass}",
        # "m_jj","jj_pt","deltaR_jj","dr_min_jj","dr_mean_jj",
        # "m_bb","bb_pt","deltaR_bb","dr_min_bb","dr_mean_bb",
        # "m_lbb","lbb_pt","deltaR_lbb","dr_min_lbb","dr_mean_lbb",
        # deterministic_seeds, normalization_weights

    },
    # produces={
    #     weights,
    #     # columns for ML inputs are set by the init function
    # },
)
def ml_inputs(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # attach coffea behavior
    events = ak.Array(events, behavior=coffea.nanoevents.methods.nanoaod.behavior)
    #events = set_ak_column(events, "BJet", ak.pad_none(events.BJet, 2))
    # name of table to place ML variables in
    ns = self.ml_namespace

    # object arrays
    jet = ak.with_name(events.Jet, "Jet")
    bjet = ak.with_name(events.BJet, "Jet")
    Electrons = ak.with_name(events.Electron, "PtEtaPhiMLorentzVector")
    Muons = ak.with_name(events.Muon, "PtEtaPhiMLorentzVector")


    # jet/fatjet multiplicities
    events = set_ak_column(events, f"{ns}.n_jet", ak.num(events.Jet.pt, axis=1), value_type=np.int32)
    events = set_ak_column(events, f"{ns}.n_bjet", ak.num(events.BJet.pt, axis=1), value_type=np.int32)
    # lepton multiplicities    
    events = set_ak_column(events, f"{ns}.n_electron", ak.num(events.Electron.pt, axis=1), value_type=np.int32)
    events = set_ak_column(events, f"{ns}.n_muon", ak.num(events.Muon.pt, axis=1), value_type=np.int32)

    # jj-features
    jj = (jet[:, 0] + jet[:, 1])
    deltaR_jj = jet[:, 0].delta_r(jet[:, 1])

    jet_pairs = ak.combinations(jet, 2)
    dr = jet_pairs[:, :, "0"].delta_r(jet_pairs[:, :, "1"])
    dr_min_jj = ak.min(dr, axis = 1)
    dr_mean_jj = ak.mean(dr, axis = 1)
    dr_mean_jj = ak.where(np.isfinite(dr_mean_jj), dr_mean_jj, EMPTY_FLOAT) 
    #from IPython import embed; embed()

    events = set_ak_column_f32(events, f"{ns}.m_jj", jj.mass)
    events = set_ak_column_f32(events, f"{ns}.jj_pt", jj.pt)
    events = set_ak_column_f32(events, f"{ns}.deltaR_jj", deltaR_jj)
    events = set_ak_column_f32(events, f"{ns}.dr_min_jj", dr_min_jj)
    events = set_ak_column_f32(events, f"{ns}.dr_mean_jj", dr_mean_jj)


    # bb-features
    bb = (bjet[:, 0] + bjet[:, 1])
    deltaR_bb = bjet[:, 0].delta_r(bjet[:, 1])
    # find delta R between leading BJets
    bb_pairs = ak.combinations(bjet, 2)
    dr_bb = bb_pairs[:, :, "0"].delta_r(bb_pairs[:, :, "1"])

    dr_min_bb = ak.min(dr_bb, axis = 1)
    dr_mean_bb = ak.mean(dr_bb, axis = 1)
    dr_mean_bb = ak.where(np.isfinite(dr_mean_bb), dr_mean_bb, EMPTY_FLOAT)

    events = set_ak_column_f32(events, f"{ns}.m_bb", bb.mass)
    events = set_ak_column_f32(events, f"{ns}.bb_pt", bb.pt)
    events = set_ak_column_f32(events, f"{ns}.deltaR_bb", deltaR_bb)
    events = set_ak_column_f32(events, f"{ns}.dr_min_bb", dr_min_bb)
    events = set_ak_column_f32(events, f"{ns}.dr_mean_bb", dr_mean_bb)

    # l_bb-features
    l = ak.where(ak.num(Electrons)>0, Electrons, Muons)
    
    # l = ak.with_name(l, "PtEtaPhiMLorentzVector")
    deltaR_lbb = bjet[:,0].delta_r(l[:,0])
    events = set_ak_column_f32(events, f"{ns}.deltaR_lbb", deltaR_lbb)
    # events = set_ak_column_f32(events, f"{ns}.lepton", l)
    # from IPython import embed; embed()
    # -- helper functions

    def set_vars(events, name, arr, n_max, attrs, default=-10.0):
        # pad to miminal length
        arr = ak.pad_none(arr, n_max)
        # extract fields
        for i, attr in itertools.product(range(1, n_max + 1), attrs):
            # print(f"{self.ml_namespace}.{name}_{attr}_{i}")
            value = ak.nan_to_none(getattr(arr[:, i - 1], attr))
            value = ak.fill_none(value, default)
            events = set_ak_column_f32(events, f"{self.ml_namespace}.{name}{i}_{attr}", value)
        return events

    # def set_vars_single(events, name, arr, attrs, default=-10.0):
    #     for attr in attrs:
    #         # print(name)
    #         # print(f"{self.ml_namespace}.{name}_{attr}")
    #         value = ak.nan_to_none(getattr(arr, attr))
    #         value = ak.fill_none(value, default)
    #         events = set_ak_column_f32(events, f"{self.ml_namespace}.{name}_{attr}", value)
    #     return events


    # AK4 jets
    events = set_vars(
        events, "Jet", jet, n_max=5,
        attrs=("pt", "eta", "phi", "mass"),
    )
    events = set_vars(
        events, "BJet", bjet, n_max=5,
        attrs=("pt", "eta", "phi", "mass"),
    )
    # Lepton
    # events = set_vars_single(
    #     events, "Electron", Electrons,
    #     attrs=("pt", "eta", "phi"),
    # )
    # events = set_vars_single(
    #     events, "Muon", Muons,
    #     attrs=("pt", "eta", "phi"),
    # )
    events = set_vars(
        events, "lepton", l, n_max=1,
        attrs=("pt", "eta", "phi"),
    )
    
    # weights
    # events = self[weights](events, **kwargs)
    #events = self[deterministic_seeds](events, **kwargs)
    #events = self[normalization_weights](events, **kwargs)
    # from IPython import embed; embed()
    return events


@ml_inputs.init
def ml_inputs_init(self: Producer) -> None:
    # put ML input columns in separate namespace/table
    self.ml_namespace = "MLInput"

    # store column names
    self.ml_columns = {
        # number of objects
        "n_jet","n_bjet",
        # "n_electron","n_muon",
        # "Electron_pt","Electron_eta","Electron_phi",
        # "Muon_pt","Muon_eta","Muon_phi",
        "lepton1_pt","lepton1_eta","lepton1_phi",
        "m_jj","jj_pt","deltaR_jj","dr_min_jj","dr_mean_jj",
        "m_bb","bb_pt","deltaR_bb","dr_min_bb","dr_mean_bb",
        "deltaR_lbb",
    } | {
        f"Jet{i + 1}_{var}"
        for var in ("pt", "eta", "phi", "mass" )
        for i in range(5)
    } | {
        f"BJet{i + 1}_{var}"
        for var in ("pt", "eta", "phi", "mass" )
        for i in range(3)
    }
    # declare produced columns
    self.produces |= {
        f"{self.ml_namespace}.{col}" for col in self.ml_columns
    }
    # self.produces|= {
    #     deterministic_seeds
    # }
    # self.produces|= {
    #     normalization_weights
    # }
    
    # add production categories to config
    if not self.config_inst.get_aux("has_categories_production", False):
        self.config_inst.x.has_categories_production = True

    # add ml variables to config
    if not self.config_inst.get_aux("has_variables_ml", False):
        # add_variables_ml(self.config_inst)
        self.config_inst.x.has_variables_ml = True