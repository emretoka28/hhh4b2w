"""
Category definition
"""
from typing import OrderedDict
import law
import order as od
from functools import wraps
from columnflow.config_util import add_category 
from columnflow.categorization import Categorizer, categorizer
from columnflow.config_util import create_category_combinations


def call_once_on_config(include_hash=False):
    """
    Parametrized decorator to ensure that function *func* is only called once for the config *config*
    """
    def outer(func):
        @wraps(func)
        def inner(config, *args, **kwargs):
            tag = f"{func.__name__}_called"
            if include_hash:
                tag += f"_{func.__hash__()}"

            if config.has_tag(tag):
                return

            config.add_tag(tag)
            return func(config, *args, **kwargs)
        return inner
    return outer

def name_fn(root_cats):
    """ define how to combine names for category combinations """
    cat_name = "__".join(cat.name for cat in root_cats.values())
    return cat_name

def kwargs_fn(root_cats):
    """ define how to combine category attributes for category combinations """
    kwargs = {
        "id": sum([c.id for c in root_cats.values()]),
        "label": ", ".join([c.name for c in root_cats.values()]),
        # optional: store the information on how the combined category has been created
        "aux": {
            "root_cats": {key: value.name for key, value in root_cats.items()},
        },
    }
    return kwargs

logger = law.logger.get_logger(__name__)

@call_once_on_config()
def add_categories(config: od.Config) -> None:
    
    jet_categories = (5,6)
    bjet_categories = (3,4)
    lep_categories=("1e", "1mu") 
    
    for lep_cat in lep_categories:
        config.add_category(
            id=10 if lep_cat == "1e" else 20,
            name=lep_cat,
            selection=f"cat_{lep_cat}",
            label=f"{lep_cat}",
        )
    for n_jet in jet_categories:
        # from IPython import embed; embed()
        if n_jet == 5:
            config.add_category(
                id=100 * n_jet,
                name=f"{n_jet}j",
                selection=f"cat_{n_jet}j",
                label=f"$\eq${n_jet} jets",
            )
        elif n_jet == 6:
            config.add_category(
                id=100 * n_jet,
                name=f"{n_jet}j",
                selection=f"cat_{n_jet}j",
                label=f"$\geq${n_jet} jets",
            )
    
    for n_bjet in bjet_categories:
        if n_bjet == 3:
            config.add_category(
                id=1000 * n_bjet,
                name=f"{n_bjet}bj",
                selection=f"cat_{n_bjet}bj",
                label=f"$\eq${n_bjet} b-tagged jets",
            )
        elif n_bjet == 4:
            config.add_category(
                id=1000 * n_bjet,
                name=f"{n_bjet}bj",
                selection=f"cat_{n_bjet}bj",
                label=f"$\geq${n_bjet} b-tagged jets",
            )

    category_blocks = OrderedDict({
         "lepton": [config.get_category(f"{lep_cat}") for lep_cat in lep_categories],
         "jet": [config.get_category(f"{n_jet}j") for n_jet in jet_categories],
         "bjet": [config.get_category(f"{n_bjet}bj") for n_bjet in bjet_categories],
     })

    # from IPython import embed; embed()
    n_cats = create_category_combinations(
        config,
        category_blocks,
        name_fn=name_fn,
        kwargs_fn=kwargs_fn,
        skip_existing=False,
    )     

    print(f"{n_cats} have been created by the create_category_combinations function")
    all_cats = [cat.name for cat, _, _ in config.walk_categories()]
    print(f"List of all cateogries in our config: \n{all_cats}")

# @call_once_on_config()
# def add_categories(config: od.Config) -> None:

#     cat_incl = config.add_category(
#         id=1,
#         name="incl",
#         selection="cat_incl",
#         label="inclusive",
#     )
#     # cat_2j = config.add_category(
#     #     id=2,
#     #     name="2j",
#     #     selection="cat_2j",
#     #     label="2 jets",
#     # )


#     # Electron/Muon categories
#     cat_1e = config.add_category(
#         name="1e",
#         id=10,
#         selection="cat_1e",
#         label="1 Electron, 0 Muons",
#     )
#     cat_1mu = config.add_category(
#         name="1mu",
#         id=20,
#         selection="cat_1mu",
#         label="1 Muon, 0 Electrons",
#     )

#     # Jet categories
#     cat_6j_4bj = config.add_category(
#         name="6j_4bj",
#         id=30,
#         selection="cat_6j_4bj",
#         label=">=6 J, >=4 B-J",
#     )
#     cat_5j_4bj = config.add_category(
#         name="5j_4bj",
#         id=31,
#         selection="cat_5j_4bj",
#         label="5 J, >=4 B-J",
#     )
#     cat_6j_3bj = config.add_category(
#         name="6j_3bj",
#         id=32,
#         selection="cat_6j_3bj",
#         label=">=6 J, 3 B-J",
#     )
#     cat_5j_3bj = config.add_category(
#         name="5j_3bj",
#         id=33,
#         selection="cat_5j_3bj",
#         label="5 J, 3 B-J",
#     )

#     # Combination of Jets + Lepton

#     cat_1e_6j_4bj = cat_6j_4bj.add_category(
#         name="1e_6j_4bj",
#         id=cat_6j_4bj.id + cat_1e.id,
#         selection=[cat_1e.selection, cat_6j_4bj.selection],
#         label="1 e\n$\\geq 6$ jets\n$\\geq 4$ b-jets"
#     )

#     cat_1mu_6j_4bj = cat_6j_4bj.add_category(
#         name="1mu_6j_4bj",
#         id=cat_6j_4bj.id + cat_1mu.id,
#         selection=[cat_1mu.selection, cat_6j_4bj.selection],
#         label="1 $\mu$\n$\\geq 6$ jets\n$\\geq 4$ b-jets",
#     )

#     cat_1e_5j_4bj = cat_5j_4bj.add_category(
#         name="1e_5j_4bj",
#         id=cat_5j_4bj.id + cat_1e.id,
#         selection=[cat_1e.selection, cat_5j_4bj.selection],
#         label="1 e\n$= 5$ jets\n$\\geq 4$ b-jets",
#     )

#     cat_1e_5j_3bj = cat_5j_3bj.add_category(
#         name="1e_5j_3bj",
#         id=cat_5j_3bj.id + cat_1e.id,
#         selection=[cat_1e.selection, cat_5j_3bj.selection],
#         label="1 e\n$= 5$ jets\n$= 3$ b-jets",
#     )

#     cat_1mu_5j_4bj = cat_5j_4bj.add_category(
#         name="1mu_5j_4bj",
#         id=cat_5j_4bj.id + cat_1mu.id,
#         selection=[cat_1mu.selection, cat_5j_4bj.selection],
#         label="1 $\mu$\n$= 5$ jets\n$\\geq 4$ b-jets",
#     )

#     cat_1e_6j_3bj = cat_6j_3bj.add_category(
#         name="1e_6j_3bj",
#         id=cat_6j_3bj.id + cat_1e.id,
#         selection=[cat_1e.selection, cat_6j_3bj.selection],
#         label="1 e\n$\\geq 6$ jets\n$= 3$ b-jets",
#     )   
    
#     cat_1mu_6j_3bj = cat_6j_3bj.add_category(
#         name="1mu_6j_3bj",
#         id=cat_6j_3bj.id + cat_1mu.id,
#         selection=[cat_1mu.selection, cat_6j_3bj.selection],
#         label="1 $\mu$\n$\\geq 6$ jets\n$= 3$ b-jets",
#     )  
    
#     cat_1mu_5j_3bj = cat_5j_3bj.add_category(
#         name="1mu_5j_3bj",
#         id=cat_5j_3bj.id + cat_1mu.id,
#         selection=[cat_1mu.selection, cat_5j_3bj.selection],
#         label="1 $\mu$\n$= 5$ jets\n$= 3$ b-jets",
#     )
    