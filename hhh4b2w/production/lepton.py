import functools
import itertools
from columnflow.production import Producer, producer
from columnflow.production.util import attach_coffea_behavior
from columnflow.selection.util import create_collections_from_masks
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column


np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses={
        "Muon.{pt,eta,phi,mass}",
        "Electron.{pt,eta,phi,mass}",

    },
    produces={
        "lepton.{pt,eta,phi,mass}",#"n_lepton","n_lepton_low_pt",
      
    },
)
def lepton_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # attach coffea behavior
    events = ak.Array(events, behavior=coffea.nanoevents.methods.nanoaod.behavior)

    Electrons = ak.with_name(events.Electron, "PtEtaPhiMLorentzVector")
    Muons = ak.with_name(events.Muon, "PtEtaPhiMLorentzVector")

    lepton = ak.where(ak.num(Electrons)>0, Electrons, Muons)
    events = set_ak_column(events, "lepton", lepton)  
    lepton = ak.with_name(lepton, "PtEtaPhiMLorentzVector")
    n_lepton = ak.num(lepton.pt, axis=1)

    lepton_mask_lower = events.lepton.pt < 30
    lepton_mask_higher = events.lepton.pt >= 30

    n_lepton_low_pt = ak.count_nonzero(lepton_mask_lower, axis=1)
    n_lepton_low_pt = ak.count_nonzero(n_lepton_low_pt) 
    n_lepton_high_pt = ak.count_nonzero(lepton_mask_higher, axis=1)
    n_lepton_high_pt = ak.count_nonzero(n_lepton_high_pt)   
    
    
    # from IPython import embed; embed()
    events = set_ak_column(events, "n_lepton", n_lepton, value_type=np.int32)
    

    return events

