from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT
from columnflow.selection import SelectionResult


ak = maybe_import("awkward")
np = maybe_import("numpy")

@producer(
    uses={
        "BJet.pt", "BJet.eta",
        #"gen_hhh4b2w_decay",
        "gen_hhh4b2w_decay_b",
        "GenPart.*"
    },
    #produces={"Bjet_matches.pt", "Bjet_nonmatches.pt"}
    produces={"BJet.matched_higgs"}
)
def gen_hhh4b2w_matching(
        self: Producer, events: ak.Array,
        results: SelectionResult = None, verbose: bool = True,
        dR_req: float = 0.4,
        **kwargs,
) -> ak.Array:
    """
    Function that matches HHH->bbbbWW decay product gen particles to Reco-level jets and leptons.
    """

    if self.dataset_inst.is_data or not self.dataset_inst.has_tag("HHH"):
        return events

    # Dictionary to hold the matches for each jet
    jet_matches = ak.Array([None] * len(events.BJet))
    # Track already matched jet indices
    used_jet_indices = set()

    for gp_tag in ("b1", "b2", "b3", "b4"):
    # Get the generator-level particle for the current b-quark
        gp = events.gen_hhh4b2w_decay_b[gp_tag]
        dr = events.BJet.delta_r(gp)
    
        # Find the closest match
        closest_idx = ak.argmin(dr, axis=-1)  # Index of closest match
        closest_dr = dr[closest_idx]         # Actual delta R value
    
        # Apply dR requirement and ensure jet is not already matched
        valid_mask = (closest_dr < dR_req) & (~ak.Array([idx in used_jet_indices for idx in ak.to_list(closest_idx)]))
    
        # Mask closest_idx using valid_mask, ensuring shapes match
        matched_jets = ak.mask(closest_idx, valid_mask)  # Properly mask closest_idx

        # If a valid match is found, mark it
        for match_idx in ak.to_list(matched_jets[~ak.is_none(matched_jets)]):  # Filter non-matching indices
            jet_matches[match_idx] = gp_tag
            used_jet_indices.add(match_idx)


    # Assign matches to the jets
    events = set_ak_column(events, "BJet.matched_b_quark", jet_matches)
    from IPython import embed; embed()
    return events


    # jet matching
    # for gp_tag in ("b1", "b2", "b3", "b4"):
    #     # split up to b1,b2 & b3,b4 
    #     # add b1/b2 to one column and b3/b4 to another and plot them in the same plot
    #     gp = events.gen_hhh4b2w_decay_b[gp_tag]
    #     dr = events.BJet.delta_r(gp)

    #     # matches
    #     jet_match_mask = (dr < dR_req)
    #     jet_matches = events.BJet[jet_match_mask]
    #     # non-matches
    #     jet_nonmatch_mask = (dr > dR_req)
    #     jet_nonmatches = events.BJet[jet_nonmatch_mask]

    #     if verbose:
    #         print(gp_tag, "multiple matches:", ak.sum(ak.num(jet_matches) >= 1))
    #         print(gp_tag, "no matches:", ak.sum(ak.num(jet_matches) == 0))

    #     # if multiple matches found: choose match with smallest dr
    #     jet_match = jet_matches[ak.argsort(jet_matches.delta_r(gp))]
    #     jet_nonmatch = jet_nonmatches[ak.argsort(jet_nonmatches.delta_r(gp))]
    #     gen_matches[gp_tag] = jet_match

    #     if gp_tag == "b1":
    #         mother_b1 = gp.genPartIdxMother    
    #     elif gp_tag == "b2":           
    #         mother_b2 = gp.genPartIdxMother 
    #     elif gp_tag == "b3":
    #         mother_b3 = gp.genPartIdxMother  
    #     elif gp_tag == "b4":        
    #         mother_b4 = gp.genPartIdxMother
      
    
    #     if gp_tag == "b1" or gp_tag == "b2":
    #         mother_tag = "h1"  # Mother particle for b1 and b2 is h1
    #         jet_matches = set_ak_column(jet_matches, "Mother of b1/b2", mother_tag)
    #     else:
    #         mother_tag = "h2"  # Mother particle for b3 and b4 is h2
    #         jet_matches = set_ak_column(jet_matches, "Mother of b3/b4", mother_tag)
    #     # Set the "Mother" attribute for each matched jet to the corresponding Higgs boson (h1 or h2)
        # b1_b2_equal = mother_b1 == mother_b2
        # b3_b4_equal = mother_b3 == mother_b4