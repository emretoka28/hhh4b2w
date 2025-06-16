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

jet_categories = (5, 6)
bjet_categories = (3,4)
lep_categories=("1e", "1mu")


for n_jet in jet_categories:
    @categorizer(cls_name=f"cat_{n_jet}j" ,uses={"Jet.pt"})
    def cat_n_jet(self:Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
        """
        Categorizer for n jets.
        """
        if n_jet == 5:
            return events, ak.num(events.Jet.pt, axis=1) == n_jet
        elif n_jet == 6:
            return events, ak.num(events.Jet.pt, axis=1) >= n_jet

for n_bjet in bjet_categories:
    @categorizer(cls_name=f"cat_{n_bjet}bj", uses={"BJet.pt"})
    def cat_n_bjet(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
        """
        Categorizer for n b-jets.
        """
        if n_bjet == 3:
            return events, ak.num(events.BJet.pt, axis=1) == n_bjet
        elif n_bjet == 4:
            return events, ak.num(events.BJet.pt, axis=1) >= n_bjet
        

for lep_cat in lep_categories:
    @categorizer(cls_name=f"cat_{lep_cat}", uses={"Muon.pt", "Electron.pt"})
    def cat_lepton(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
        """
        Categorizer for lepton categories.
        """
        if lep_cat == "1e":
            return events, (ak.num(events.Muon.pt, axis=1) == 0) & (ak.num(events.Electron.pt, axis=1) == 1)
        elif lep_cat == "1mu":
            return events, (ak.num(events.Muon.pt, axis=1) == 1) & (ak.num(events.Electron.pt, axis=1) == 0)


# # Lepton categorizer
# @categorizer(uses={"Muon.pt", "Electron.pt"})
# def cat_1e(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
#     return events, ((ak.num(events.Muon.pt, axis=1) == 0) & (ak.num(events.Electron.pt, axis=1) == 1))

# @categorizer(uses={"Muon.pt", "Electron.pt"})
# def cat_1mu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
#     return events, ((ak.num(events.Muon.pt, axis=1) == 1) & (ak.num(events.Electron.pt, axis=1) == 0))

# # Jet categorizer
# @categorizer(uses={"Jet.pt"})
# def cat_5j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
#     return events, ak.num(events.Jet.pt, axis=1) == 5
# @categorizer(uses={"Jet.pt"})
# def cat_6j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
#     return events, ak.num(events.Jet.pt, axis=1) >= 6

# # BJet categorizer
# @categorizer(uses={"BJet.pt"})
# def cat_3bj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
#     return events, ak.num(events.BJet.pt, axis=1) == 3
# @categorizer(uses={"BJet.pt"})
# def cat_4bj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:        
#     return events, ak.num(events.BJet.pt, axis=1) >= 4







# @categorizer(uses={"Jet.pt", "BJet.pt"})
# def cat_6j_4bj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
#     return events, ((ak.num(events.Jet.pt, axis=1) >= 6) & (ak.num(events.BJet.pt, axis=1) >= 4))

# @categorizer(uses={"Jet.pt", "BJet.pt"})
# def cat_5j_4bj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
#     return events, ((ak.num(events.Jet.pt, axis=1) == 5) & (ak.num(events.BJet.pt, axis=1) >= 4))

# @categorizer(uses={"Jet.pt", "BJet.pt"})
# def cat_6j_3bj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
#     return events, ((ak.num(events.Jet.pt, axis=1) >= 6) & (ak.num(events.BJet.pt, axis=1) == 3))

# @categorizer(uses={"Jet.pt", "BJet.pt"})
# def cat_5j_3bj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
#     return events, ((ak.num(events.Jet.pt, axis=1) == 5) & (ak.num(events.BJet.pt, axis=1) == 3))


# # Lepton + Jet configurations
# @categorizer(uses={"Muon.pt", "Electron.pt", "Jet.pt", "BJet.pt"})
# def cat_1e_6j_4bj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
#     return events, ((ak.num(events.Muon.pt, axis=1) == 0) & (ak.num(events.Electron.pt, axis=1) == 1) 
#                     & (ak.num(events.Jet.pt, axis=1) >= 6) & (ak.num(events.BJet.pt, axis=1) >= 4))

# @categorizer(uses={"Muon.pt", "Electron.pt", "Jet.pt", "BJet.pt"})
# def cat_1e_5j_4bj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
#     return events, ((ak.num(events.Muon.pt, axis=1) == 0) & (ak.num(events.Electron.pt, axis=1) == 1) 
#                     & (ak.num(events.Jet.pt, axis=1) == 5) & (ak.num(events.BJet.pt, axis=1) >= 4))

# @categorizer(uses={"Muon.pt", "Electron.pt", "Jet.pt", "BJet.pt"})
# def cat_1e_6j_3bj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
#     return events, ((ak.num(events.Muon.pt, axis=1) == 0) & (ak.num(events.Electron.pt, axis=1) == 1) 
#                     & (ak.num(events.Jet.pt, axis=1) >= 6) & (ak.num(events.BJet.pt, axis=1) == 3))

# @categorizer(uses={"Muon.pt", "Electron.pt", "Jet.pt", "BJet.pt"})
# def cat_1e_5j_3bj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
#     return events, ((ak.num(events.Muon.pt, axis=1) == 0) & (ak.num(events.Electron.pt, axis=1) == 1) 
#                     & (ak.num(events.Jet.pt, axis=1) == 5) & (ak.num(events.BJet.pt, axis=1) == 3))


# @categorizer(uses={"Muon.pt", "Electron.pt", "Jet.pt", "BJet.pt"})
# def cat_1mu_6j_4bj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
#     return events, ((ak.num(events.Muon.pt, axis=1) == 1) & (ak.num(events.Electron.pt, axis=1) == 0) 
#                     & (ak.num(events.Jet.pt, axis=1) >= 6) & (ak.num(events.BJet.pt, axis=1) >= 4))

# @categorizer(uses={"Muon.pt", "Electron.pt", "Jet.pt", "BJet.pt"})
# def cat_1mu_5j_4bj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
#     return events, ((ak.num(events.Muon.pt, axis=1) == 1) & (ak.num(events.Electron.pt, axis=1) == 0) 
#                     & (ak.num(events.Jet.pt, axis=1) == 5) & (ak.num(events.BJet.pt, axis=1) >= 4))

# @categorizer(uses={"Muon.pt", "Electron.pt", "Jet.pt", "BJet.pt"})
# def cat_1mu_6j_3bj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
#     return events, ((ak.num(events.Muon.pt, axis=1) == 1) & (ak.num(events.Electron.pt, axis=1) == 0)
#                     & (ak.num(events.Jet.pt, axis=1) >= 6) & (ak.num(events.BJet.pt, axis=1) == 3))

# @categorizer(uses={"Muon.pt", "Electron.pt", "Jet.pt", "BJet.pt"})
# def cat_1mu_5j_3bj(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
#     return events, ((ak.num(events.Muon.pt, axis=1) == 1) & (ak.num(events.Electron.pt, axis=1) == 0) 
#                     & (ak.num(events.Jet.pt, axis=1) == 5) & (ak.num(events.BJet.pt, axis=1) == 3))


# @categorizer(uses={})

