# coding: utf-8

"""
Exemplary selection methods.
"""

from columnflow.categorization import Categorizer, categorizer
from columnflow.util import maybe_import

ak = maybe_import("awkward")


#
# categorizer functions used by categories definitions
#
# Examples
@categorizer(uses={"event"})
def cat_incl(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # fully inclusive selection
    return events, ak.ones_like(events.event) == 1


@categorizer(uses={"Jet.pt"})
def cat_2j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # two or more jets
    return events, ak.num(events.Jet.pt, axis=1) >= 2

# Lepton categorizer
@categorizer(uses={"Muon.pt", "Electron.pt"})
def cat_1e(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, ((ak.num(events.Muon.pt, axis=1) == 0) & (ak.num(events.Electron.pt, axis=1) == 1))

@categorizer(uses={"Muon.pt", "Electron.pt"})
def cat_1mu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, ((ak.num(events.Muon.pt, axis=1) == 1) & (ak.num(events.Electron.pt, axis=1) == 0))


# Jet categorizer
@categorizer(uses={"Jet.pt", "BJet.pt"})
def cat_6j_4bj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, ((ak.num(events.Jet.pt, axis=1) >= 6) & (ak.num(events.BJet.pt, axis=1) >= 4))

@categorizer(uses={"Jet.pt", "BJet.pt"})
def cat_5j_4bj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, ((ak.num(events.Jet.pt, axis=1) >= 5) & (ak.num(events.BJet.pt, axis=1) >= 4))

@categorizer(uses={"Jet.pt", "BJet.pt"})
def cat_6j_3bj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, ((ak.num(events.Jet.pt, axis=1) >= 6) & (ak.num(events.BJet.pt, axis=1) >= 3))


# Lepton + Jet configurations
@categorizer(uses={"Muon.pt", "Electron.pt", "Jet.pt", "BJet.pt"})
def cat_1e_6j_4bj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, ((ak.num(events.Muon.pt, axis=1) == 0) & (ak.num(events.Electron.pt, axis=1) == 1) 
                    & (ak.num(events.Jet.pt, axis=1) >= 6) & (ak.num(events.BJet.pt, axis=1) >= 4))

@categorizer(uses={"Muon.pt", "Electron.pt", "Jet.pt", "BJet.pt"})
def cat_1e_5j_4bj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, ((ak.num(events.Muon.pt, axis=1) == 0) & (ak.num(events.Electron.pt, axis=1) == 1) 
                    & (ak.num(events.Jet.pt, axis=1) >= 5) & (ak.num(events.BJet.pt, axis=1) >= 4))

@categorizer(uses={"Muon.pt", "Electron.pt", "Jet.pt", "BJet.pt"})
def cat_1e_6j_3bj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, ((ak.num(events.Muon.pt, axis=1) == 0) & (ak.num(events.Electron.pt, axis=1) == 1) 
                    & (ak.num(events.Jet.pt, axis=1) >= 6) & (ak.num(events.BJet.pt, axis=1) >= 3))


@categorizer(uses={"Muon.pt", "Electron.pt", "Jet.pt", "BJet.pt"})
def cat_1mu_6j_4bj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, ((ak.num(events.Muon.pt, axis=1) == 1) & (ak.num(events.Electron.pt, axis=1) == 0) 
                    & (ak.num(events.Jet.pt, axis=1) >= 6) & (ak.num(events.BJet.pt, axis=1) >= 4))

@categorizer(uses={"Muon.pt", "Electron.pt", "Jet.pt", "BJet.pt"})
def cat_1mu_5j_4bj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, ((ak.num(events.Muon.pt, axis=1) == 1) & (ak.num(events.Electron.pt, axis=1) == 0) 
                    & (ak.num(events.Jet.pt, axis=1) >= 5) & (ak.num(events.BJet.pt, axis=1) >= 4))

@categorizer(uses={"Muon.pt", "Electron.pt", "Jet.pt", "BJet.pt"})
def cat_1mu_6j_3bj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, ((ak.num(events.Muon.pt, axis=1) == 1) & (ak.num(events.Electron.pt, axis=1) == 0)
                    & (ak.num(events.Jet.pt, axis=1) >= 6) & (ak.num(events.BJet.pt, axis=1) >= 3))




