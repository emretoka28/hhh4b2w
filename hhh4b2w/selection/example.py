# coding: utf-8

"""
Exemplary selection methods.
"""

from collections import defaultdict

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.stats import increment_stats
from columnflow.selection.util import sorted_indices_from_mask
from columnflow.production.processes import process_ids
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.util import maybe_import

from hhh4b2w.production.example import cutflow_features, gen_hhh4b2w_decay_products

from columnflow.production.util import attach_coffea_behavior
from typing import Tuple

from columnflow.columnar_util import set_ak_column

np = maybe_import("numpy")
ak = maybe_import("awkward")


def masked_sorted_indices(mask: ak.Array, sort_var: ak.Array, ascending: bool = False) -> ak.Array:
    """
    Helper function to obtain the correct indices of an object mask
    """
    indices = ak.argsort(sort_var, axis=-1, ascending=ascending)
    return indices[mask[indices]]


@selector(
        uses={"Electron.pt", "Electron.eta","Muon.pt", "Muon.eta","Electron.mvaIso_WP80","Muon.highPtId","Muon.tkIsoId"},
)
def lepton_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    # example electron selection: exactly one muon
    electron_mask = (events.Electron.pt >= 25.0) & (abs(events.Electron.eta) < 2.4) & (events.Electron.mvaIso_WP80)
    muon_mask = (events.Muon.pt >= 25.0) & (abs(events.Muon.eta) < 2.4) & (events.Muon.highPtId == 2) & (events.Muon.tkIsoId == 2)

    # muon: tightId/pfIsoId

    lepton_sel = (ak.sum(electron_mask, axis=1) == 1) | (ak.sum(muon_mask, axis=1) == 1)

    # build and return selection results
    # "objects" maps source columns to new columns and selections to be applied on the old columns
    # to create them, e.g. {"Muon": {"MySelectedMuon": indices_applied_to_Muon}}
    return events, SelectionResult(
        steps={
            "lepton": lepton_sel,
        },
        objects={
            "Electron": {
                "Electron": electron_mask,
            },
            "Muon":{
                "Muon": muon_mask,
            }
        },
        aux={
            "n_electrons": ak.num(events.Electron.pt, axis = 1),
            "n_muons": ak.sum(events.Muon.pt, axis=1)
        }
    )



@selector(
    uses={"Jet.pt", "Jet.eta"},
)
def jet_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    # example jet selection: at least one jet
    jet_mask = (events.Jet.pt >= 25.0) & (abs(events.Jet.eta) < 2.4)
    jet_sel = ak.sum(jet_mask, axis=1) >= 5

    # build and return selection results
    # "objects" maps source columns to new columns and selections to be applied on the old columns
    # to create them, e.g. {"Jet": {"MyCustomJetCollection": indices_applied_to_Jet}}
    return events, SelectionResult(
        steps={
            "jet": jet_sel,
        },
        objects={
            "Jet": {
                "Jet": sorted_indices_from_mask(jet_mask, events.Jet.pt, ascending=False),
            },
        },
        aux={
            "n_jets": ak.sum(jet_mask, axis=1),
        },
    )


@selector(
    uses={"Jet.pt", "Jet.eta", "Jet.phi", "Jet.jetId", "Jet.btagDeepFlavB"},
    exposed=True,
)
def bjet_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    # DiJet jet selection
    # - require ...

    # assign local index to all Jets - stored after masks for matching
    # TODO: Drop for dijet ?
    events = set_ak_column(events, "Jet.local_index", ak.local_index(events.Jet))

    # jets
    # TODO: Correct jets
    # Selection by UHH2 framework
    # https://github.com/UHH2/DiJetJERC/blob/ff98eebbd44931beb016c36327ab174fdf11a83f/src/AnalysisModule_DiJetTrg.cxx#L692
    # IDs in NanoAOD https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD
    #  & JME NanoAOD https://cms-nanoaod-integration.web.cern.ch/integration/master-106X/mc102X_doc.html
    jet_mask = (
        (events.Jet.pt > 25) &
        (abs(events.Jet.eta) < 2.4) &
        # IDs in NanoAOD https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD
        (events.Jet.jetId == 6) |   # 2: fail tight LepVeto and 6: pass tightLepVeto
        (events.Jet.pt > 50)  # pass all IDs (l, m and t) only for jets with pt < 50 GeV
    )

    jet_sel = ak.num(events.Jet[jet_mask]) >= 5

    # btagging
    wp_med = self.config_inst.x.btag_working_points.deepjet.medium
    bjet_mask = jet_mask & (events.Jet.btagDeepFlavB >= wp_med)
    bjet_sel = ak.num(events.Jet[bjet_mask]) >= 3

    jet_indices = masked_sorted_indices(jet_mask, events.Jet.pt)
    bjet_indices = masked_sorted_indices(bjet_mask, events.Jet.pt)
    jet_sel = ak.fill_none(jet_sel, False)
    bjet_sel = ak.fill_none(bjet_sel, False)
    jet_mask = ak.fill_none(jet_mask, False)
    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={
            "BJet": bjet_sel,
        },
        objects={
            "Jet": {
                "Jet": jet_indices,
                "BJet": bjet_indices,
            },
        },
        aux={
            "jet_mask": jet_mask,
            "n_central_jets": ak.num(jet_indices),
            "n_bjets": ak.sum(bjet_mask),
        },
    )


# MAIN SELECTOR

@selector(
    uses={
        # selectors / producers called within _this_ selector
        mc_weight, cutflow_features, process_ids, jet_selection, lepton_selection,
        increment_stats, gen_hhh4b2w_decay_products, attach_coffea_behavior, bjet_selection,
    },
    produces={
        # selectors / producers whose newly created columns should be kept
        mc_weight, cutflow_features, process_ids, gen_hhh4b2w_decay_products, attach_coffea_behavior,
    },
    exposed=True,
)
def example(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # # muon selection
    # events, muon_results = self[muon_selection](events, **kwargs)
    # results += muon_results

    # #electron selection
    # events, electron_results = self[electron_selection](events, **kwargs)
    # results += electron_results   

    events, lepton_results = self[lepton_selection](events, **kwargs)
    results += lepton_results   

    # jet selection
    events, jet_results = self[jet_selection](events, **kwargs)
    results += jet_results

    # b-tag jet selection
    events, bjet_results = self[bjet_selection](events, **kwargs)
    results += bjet_results

    # combined event selection after all steps
    results.event = results.steps.lepton & results.steps.jet & results.steps.BJet
    results.steps["empty"] = ak.ones_like(events.event) == 1

    # create process ids
    events = self[process_ids](events, **kwargs)

    # add the mc weight
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    # add cutflow features, passing per-object masks
    events = self[cutflow_features](events, results.objects, **kwargs)
    events = self[attach_coffea_behavior](events, **kwargs)
    events = self[gen_hhh4b2w_decay_products](events, **kwargs)

    # increment stats
    weight_map = {
        "num_events": Ellipsis,
        "num_events_selected": results.event,
    }
    group_map = {}
    if self.dataset_inst.is_mc:
        weight_map = {
            **weight_map,
            # mc weight for all events
            "sum_mc_weight": (events.mc_weight, Ellipsis),
            "sum_mc_weight_selected": (events.mc_weight, results.event),
        }
        group_map = {
            # per process
            "process": {
                "values": events.process_id,
                "mask_fn": (lambda v: events.process_id == v),
            },
            # per jet multiplicity
            "njet": {
                "values": results.x.n_jets,
                "mask_fn": (lambda v: results.x.n_jets == v),
            },
        }
    events, results = self[increment_stats](
        events,
        results,
        stats,
        weight_map=weight_map,
        group_map=group_map,
        **kwargs,
    )

    return events, results
