from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.normalization import normalization_weights
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.muon import muon_weights
from columnflow.selection.util import create_collections_from_masks
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column
import numpy as np
import functools


np = maybe_import("numpy")
ak = maybe_import("awkward")
maybe_import("coffea.nanoevents.methods.nanoaod")

set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)

@producer(
    uses={"Jet.{pt,eta,phi,mass}"},
    produces={"m_jj", "jj_pt", "deltaR_jj","dr_min_jj", "dr_mean_jj"},
)
def jj_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # create jj features
    events = set_ak_column(events, "Jet", ak.pad_none(events.Jet, 2))
    jj = (events.Jet[:, 0] + events.Jet[:, 1])
    deltaR_jj = events.Jet[:, 0].delta_r(events.Jet[:, 1])

    jet_pairs = ak.combinations(events.Jet, 2)
    dr = jet_pairs[:, :, "0"].delta_r(jet_pairs[:, :, "1"])
    dr_min_jj = ak.min(dr, axis = 1)
    dr_mean_jj = ak.mean(dr, axis = 1)
    dr_mean_jj = ak.where(np.isfinite(dr_mean_jj), dr_mean_jj, EMPTY_FLOAT) 

    

    events = set_ak_column_f32(events, "m_jj", jj.mass)
    events = set_ak_column_f32(events, "jj_pt", jj.pt)
    events = set_ak_column_f32(events, "deltaR_jj", deltaR_jj)
    events = set_ak_column_f32(events,"dr_min_jj", dr_min_jj)
    events = set_ak_column_f32(events,"dr_mean_jj", dr_mean_jj)
    
    # fill none values
    for col in self.produces:
        events = set_ak_column_f32(events, col, ak.fill_none(events[col], EMPTY_FLOAT))
    return events

@producer(
    uses={"BJet.{pt,eta,phi,mass}"},
    produces={"m_bb", "bb_pt", "deltaR_bb", "dr_min_bb", "dr_mean_bb"},
)
def bb_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # create bb features
    events = set_ak_column(events, "BJet", ak.pad_none(events.BJet, 2))
    bb = (events.BJet[:, 0] + events.BJet[:, 1])
    deltaR_bb = events.BJet[:, 0].delta_r(events.BJet[:, 1])
    # find delta R between leading BJets
    bb_pairs = ak.combinations(events.BJet, 2)
    dr_bb = bb_pairs[:, :, "0"].delta_r(bb_pairs[:, :, "1"])
    #from IPython import embed; embed()
    dr_min_bb = ak.min(dr_bb, axis = 1)
    dr_mean_bb = ak.mean(dr_bb, axis = 1)
    dr_mean_bb = ak.where(np.isfinite(dr_mean_bb), dr_mean_bb, EMPTY_FLOAT)

    events = set_ak_column_f32(events, "m_bb", bb.mass)
    events = set_ak_column_f32(events, "bb_pt", bb.pt)
    events = set_ak_column_f32(events, "deltaR_bb", deltaR_bb)
    events = set_ak_column_f32(events, "dr_min_bb", dr_min_bb)
    events = set_ak_column_f32(events, "dr_mean_bb", dr_mean_bb)

    # fill none values
    for col in self.produces:
        events = set_ak_column_f32(events, col, ak.fill_none(events[col], EMPTY_FLOAT))
    return events



@producer(
    uses={
        "BJet.{pt,eta,phi,mass}",
        "Electron.{pt,eta,phi,mass}",
        "Muon.{pt,eta,phi,mass}",
    },
    produces={"deltaR_lbb", "lepton.{pt,eta,phi,mass}"},
)
def l_bb_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # combine lepton and bb features 
    events = set_ak_column_f32(events, "BJet", ak.pad_none(events.BJet, 2))
    l = ak.where(ak.num(events.Electron)>0, events.Electron, events.Muon)
    deltaR_lbb = events.BJet[:,0].delta_r(l[:,0])
    events = set_ak_column_f32(events, "lepton", ak.with_name(l, "PtEtaPhiMLorentzVector"))
    events = set_ak_column_f32(events, "deltaR_lbb", deltaR_lbb)
    # from IPython import embed; embed()
    return events
