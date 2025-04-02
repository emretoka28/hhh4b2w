# coding: utf-8

"""
Column production methods related to higher-level features.
"""


from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.normalization import normalization_weights
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.muon import muon_weights
from columnflow.production.util import attach_coffea_behavior
from columnflow.selection.util import create_collections_from_masks
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column
from hhh4b2w.production.sensitive_variables import jj_features, bb_features, l_bb_features
from hhh4b2w.production.gen_bmatch import gen_hhh4b2w_matching
from hhh4b2w.production.ml_inputs import ml_inputs

np = maybe_import("numpy")
ak = maybe_import("awkward")
maybe_import("coffea.nanoevents.methods.nanoaod")


custom_collections = {
    "BJet": {
        "type_name": "Jet",
        "check_attr": "metric_table",
        "skip_fields": "*Idx*G",
    },
}

muon_id_weights = muon_weights.derive("muon_id_weights", cls_dict={
    "weight_name": "muon_id_weight",
    "get_muon_config": (lambda self: self.config_inst.x.muon_sf_id_names),
})

muon_iso_weights = muon_weights.derive("muon_iso_weights", cls_dict={
    "weight_name": "muon_iso_weight",
    "get_muon_config": (lambda self: self.config_inst.x.muon_sf_iso_names),
})

@producer(
    uses={
        mc_weight, category_ids,
        # nano columns
        "Jet.pt",
    },
    produces={
        mc_weight, category_ids,
        # new columns
        "cutflow.jet1_pt",
    },
)
def cutflow_features(
    self: Producer,
    events: ak.Array,
    object_masks: dict[str, dict[str, ak.Array]],  
    **kwargs,
) -> ak.Array: 
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    # apply object masks and create new collections
    reduced_events = create_collections_from_masks(events, object_masks)

    # create category ids per event and add categories back to the
    events = self[category_ids](reduced_events, target_events=events, **kwargs)

    # add cutflow columns
    events = set_ak_column(
        events,
        "cutflow.jet1_pt",
        Route("Jet.pt[:,0]").apply(events, EMPTY_FLOAT),
    )

    return events


@producer
def gen_hhh4b2w_decay_products(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Creates column 'gen_hhh4b2w', which includes the most relevant particles of a HHH->bbbbbb decay.
    All sub-fields correspond to individual GenParticles with fields pt, eta, phi, mass and pdgId.
    """

    if self.dataset_inst.is_data or not self.dataset_inst.has_tag("hhh4b2w"):
        return events

    # for quick checks
    def all_or_raise(arr, msg):
        if not ak.all(arr):
            raise Exception(f"{msg} in {100 * ak.mean(~arr):.3f}% of cases")

    # TODO: for now, this only works for HH->bbWW(qqlnu), but could maybe be generalized to all HH->bbWW decays

    # only consider hard process genparticles
    gp = events.GenPart
    gp["index"] = ak.local_index(gp, axis=1)
    gp = gp[events.GenPart.hasFlags("isHardProcess")]
    gp = gp[~ak.is_none(gp, axis=1)]
    abs_id = abs(gp.pdgId)
    
    # find initial-state particles
    isp = gp[ak.is_none(gp.parent.pdgId, axis=1)]

    # find all non-Higgs daughter particles from inital state
    sec = ak.flatten(isp.children, axis=2)
    sec = sec[abs(sec.pdgId) != 25]
    sec = ak.pad_none(sec, 2)  # TODO: Not all initial particles are gluons
    gp_ghost = ak.zip({f: EMPTY_FLOAT for f in sec.fields}, with_name="GenParticle")  # TODO: avoid union type
    sec = ak.fill_none(sec, gp_ghost, axis=1)  # axis=1 necessary

    # find hard Higgs bosons
    h = gp[abs_id == 25]
    nh = ak.num(h, axis=1)
    all_or_raise(nh == 3, "number of Higgs != 3")

    # bottoms from H decay
    b = gp[abs_id == 5]
    b = b[(abs(b.distinctParent.pdgId) == 25)]
    b = b[~ak.is_none(b, axis=1)]
    nb = ak.num(b, axis=1)
    all_or_raise(nb == 4, "number of bottom quarks from Higgs decay != 4")

    # Ws from H decay
    w = gp[abs_id == 24]
    w = w[(abs(w.distinctParent.pdgId) == 25)]
    w = w[~ak.is_none(w, axis=1)]
    nw = ak.num(w, axis=1)
    all_or_raise(nw == 2, "number of Ws != 2")

    # non-top quarks from W decays
    qs = gp[(abs_id >= 1) & (abs_id <= 5)]
    qs = qs[(abs(qs.distinctParent.pdgId) == 24)]
    qs = qs[~ak.is_none(qs, axis=1)]
    nqs = ak.num(qs, axis=1)
    all_or_raise((nqs % 2) == 0, "number of quarks from W decays is not dividable by 2")
    all_or_raise(nqs == 2, "number of quarks from W decays != 2")

    # leptons from W decays
    ls = gp[(abs_id >= 11) & (abs_id <= 16)]
    ls = ls[(abs(ls.distinctParent.pdgId) == 24)]
    ls = ls[~ak.is_none(ls, axis=1)]
    nls = ak.num(ls, axis=1)
    all_or_raise((nls % 2) == 0, "number of leptons from W decays is not dividable by 2")
    all_or_raise(nls == 2, "number of leptons from W decays != 2")

    all_or_raise(nqs + nls == 2 * nw, "number of W decay products invalid")

    # check if decay product charges are valid
    sign = lambda part: (part.pdgId > 0) * 2 - 1
    all_or_raise(ak.sum(sign(b), axis=1) == 0, "two ss bottoms")
    # all_or_raise(ak.sum(sign(w), axis=1) == 0, "two ss Ws")
    # all_or_raise(ak.sum(sign(qs), axis=1) == 0, "sign-imbalance for quarks")
    # all_or_raise(ak.sum(sign(ls), axis=1) == 0, "sign-imbalance for leptons")

    # identify decay products of W's
    lepton = ls[abs(ls.pdgId) % 2 == 1][:, 0]
    neutrino = ls[abs(ls.pdgId) % 2 == 0][:, 0]
    q_dtype = qs[abs(qs.pdgId) % 2 == 1][:, 0]
    q_utype = qs[abs(qs.pdgId) % 2 == 0][:, 0]

    # identify the leptonically and hadronically decaying W
    wlep = w[sign(w) == sign(lepton)][:, 0]
    whad = w[sign(w) != sign(lepton)][:, 0]

    # identify b1 as particle, b2 as antiparticle
    b1 = b[sign(b) == 1][:, 0]
    b2 = b[sign(b) == -1][:, 0]
    b3 = b[sign(b) == 1][:, 1]
    b4 = b[sign(b) == -1][:, 1]
    

    # TODO: identify H->bb and H->WW and switch from h1/h2 to hbb/hww
    # TODO: most fields have type='len(events) * ?genParticle' -> get rid of the '?'

    hhhgen = {
        # lepton = ak.with_name(lepton, "PtEtaPhiMLorentzVector"),
        "hhh": ak.with_name(h[:, 0] + h[:, 1] + h[:, 2], "LorentzVector"),
        # "hhh": h[:, 0] + h[:, 1] + h[:, 2],
        "h1": ak.with_name(h[:, 0], "PtEtaPhiMLorentzVector"),
        "h2": ak.with_name(h[:, 1], "PtEtaPhiMLorentzVector"),
        "h3": ak.with_name(h[:, 2], "PtEtaPhiMLorentzVector"),
        "b1": b1,
        "b2": b2,
        "b3": b3,
        "b4": b4,
        "wlep": wlep,
        "whad": whad,
        "l": lepton,
        "nu": neutrino,
        "q1": q_dtype,
        "q2": q_utype,
        # "sec1": sec[:, 0],
        # "sec2": sec[:, 1],
    }
    
    # add attribute for motherId
    gen_b = {
        "b1": b1,
        "b2": b2,
        "b3": b3,
        "b4": b4,
    }

    gen_hhh4b2w_decay = ak.Array({
        gp: {f: np.float32(getattr(hhhgen[gp], f)) for f in ["pt", "eta", "phi", "mass",]} for gp in hhhgen.keys()
    })

    # only for b's
    gen_hhh4b2w_decay_b = ak.Array({
        gp: {f: np.float32(getattr(gen_b[gp], f)) for f in ["pt", "eta", "phi", "mass", "genPartIdxMother"]} for gp in gen_b.keys()
    })

    # from IPython import embed; embed()
    events = set_ak_column(events, "gen_hhh4b2w_decay", gen_hhh4b2w_decay)
    events = set_ak_column(events, "gen_hhh4b2w_decay_b", gen_hhh4b2w_decay_b)
    return events


@gen_hhh4b2w_decay_products.init
def gen_hhh4b2w_decay_products_init(self: Producer) -> None:
    """
    Ammends the set of used and produced columns of :py:class:`gen_hhh4b2w_decay_products` in case
    a dataset including top decays is processed.
    """
    if getattr(self, "dataset_inst", None) and self.dataset_inst.has_tag("hhh4b2w"):
        self.uses |= {"GenPart.*"}
        self.produces |= {
            f"gen_hhh4b2w_decay.{gp}.{var}"
            for gp in ("hhh", "h1", "h2", "h3", "b1", "b2", "b3", "b4", "wlep", "whad", "l", "nu", "q1", "q2")
            for var in ("pt", "eta", "phi", "mass")
        }
        self.produces |= {
            f"gen_hhh4b2w_decay_b.{gp}.{var}"
            for gp in ("b1", "b2", "b3", "b4")
            for var in ("pt", "eta", "phi", "mass","genPartIdxMother")
        }
        # self.produces |= {"gen_hhh4b2w_decay.{hhh,h1,h2,h3}.{pt,eta,phi,mass}"}


@producer(
    uses={
        # nano columns
        "Jet.pt","BJet.pt","Electron.pt","Muon.pt",jj_features, bb_features, l_bb_features,gen_hhh4b2w_matching,
    },
    produces={
        # new columns
        "ht", "n_jet", "n_bjet", "n_electron", "n_muon",jj_features, bb_features, l_bb_features, gen_hhh4b2w_matching,
    },
)
def features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:  
    events = set_ak_column(events, "ht", ak.sum(events.Jet.pt, axis=1))
    events = set_ak_column(events, "n_jet", ak.num(events.Jet.pt, axis=1), value_type=np.int32)
    events = set_ak_column(events, "n_bjet", ak.num(events.BJet.pt, axis=1), value_type=np.int32)

    events = set_ak_column(events, "n_electron", ak.num(events.Electron.pt, axis=1), value_type=np.int32)
    events = set_ak_column(events, "n_muon", ak.num(events.Muon.pt, axis=1), value_type=np.int32)
    
    events = self[jj_features](events, **kwargs)
    events = self[bb_features](events, **kwargs)
    events = self[l_bb_features](events, **kwargs)
    # events = self[gen_hhh4b2w_decay_products](events, **kwargs)
    # events = self[gen_hhh4b2w_matching](events, **kwargs)

    return events

@producer(
    uses={
        attach_coffea_behavior, features, category_ids, normalization_weights, muon_iso_weights, muon_id_weights, deterministic_seeds, ml_inputs
    },
    produces={
        attach_coffea_behavior, features, category_ids, normalization_weights, muon_iso_weights, muon_id_weights, deterministic_seeds, ml_inputs
    },
)
def ml_producer(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections=custom_collections, **kwargs)
    # features
    events = self[features](events, **kwargs)

    # category ids
    events = self[category_ids](events, **kwargs)

    # deterministic seeds
    events = self[deterministic_seeds](events, **kwargs)
    
    #ML Inputs
    events = self[ml_inputs](events, **kwargs)

    # mc-only weights
    if self.dataset_inst.is_mc:
        # normalization weights
        events = self[normalization_weights](events, **kwargs)

        # muon weights
        events = self[muon_iso_weights](events, **kwargs)
        events = self[muon_id_weights](events, **kwargs)
    # from IPython import embed; embed()
    return events
