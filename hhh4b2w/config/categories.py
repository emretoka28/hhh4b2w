"""
Category definition
"""
import law
import order as od
from functools import wraps
from columnflow.config_util import add_category 
from columnflow.categorization import Categorizer, categorizer


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


logger = law.logger.get_logger(__name__)


@call_once_on_config()
def add_categories(config: od.Config) -> None:

    cat_incl = config.add_category(
        id=1,
        name="incl",
        selection="cat_incl",
        label="inclusive",
    )
    # cat_2j = config.add_category(
    #     id=2,
    #     name="2j",
    #     selection="cat_2j",
    #     label="2 jets",
    # )


    # Electron/Muon categories
    cat_1e = config.add_category(
        name="1e",
        id=10,
        selection="cat_1e",
        label="1 Electron, 0 Muons",
    )
    cat_1mu = config.add_category(
        name="1mu",
        id=20,
        selection="cat_1mu",
        label="1 Muon, 0 Electrons",
    )

    # Jet categories
    cat_6j_4bj = config.add_category(
        name="6j_4bj",
        id=30,
        selection="cat_6j_4bj",
        label=">=6 J, >=4 B-J",
    )
    cat_5j_4bj = config.add_category(
        name="5j_4bj",
        id=31,
        selection="cat_5j_4bj",
        label="5 J, >=4 B-J",
    )
    cat_6j_3bj = config.add_category(
        name="6j_3bj",
        id=32,
        selection="cat_6j_3bj",
        label=">=6 J, 3 B-J",
    )
    cat_5j_3bj = config.add_category(
        name="5j_3bj",
        id=33,
        selection="cat_5j_3bj",
        label="5 J, 3 B-J",
    )

    # Combination of Jets + Lepton

    cat_1e_6j_4bj = cat_6j_4bj.add_category(
        name="1e_6j_4bj",
        id=cat_6j_4bj.id + cat_1e.id,
        selection=[cat_1e.selection, cat_6j_4bj.selection],
        label="1 e\n$\\geq 6$ jets\n$\\geq 4$ b-jets"
    )

    cat_1mu_6j_4bj = cat_6j_4bj.add_category(
        name="1mu_6j_4bj",
        id=cat_6j_4bj.id + cat_1mu.id,
        selection=[cat_1mu.selection, cat_6j_4bj.selection],
        label="1 $\mu$\n$\\geq 6$ jets\n$\\geq 4$ b-jets",
    )

    cat_1e_5j_4bj = cat_5j_4bj.add_category(
        name="1e_5j_4bj",
        id=cat_5j_4bj.id + cat_1e.id,
        selection=[cat_1e.selection, cat_5j_4bj.selection],
        label="1 e\n$= 5$ jets\n$\\geq 4$ b-jets",
    )

    cat_1e_5j_3bj = cat_5j_3bj.add_category(
        name="1e_5j_3bj",
        id=cat_5j_3bj.id + cat_1e.id,
        selection=[cat_1e.selection, cat_5j_3bj.selection],
        label="1 e\n$= 5$ jets\n$= 3$ b-jets",
    )

    cat_1mu_5j_4bj = cat_5j_4bj.add_category(
        name="1mu_5j_4bj",
        id=cat_5j_4bj.id + cat_1mu.id,
        selection=[cat_1mu.selection, cat_5j_4bj.selection],
        label="1 $\mu$\n$= 5$ jets\n$\\geq 4$ b-jets",
    )

    cat_1e_6j_3bj = cat_6j_3bj.add_category(
        name="1e_6j_3bj",
        id=cat_6j_3bj.id + cat_1e.id,
        selection=[cat_1e.selection, cat_6j_3bj.selection],
        label="1 e\n$\\geq 6$ jets\n$= 3$ b-jets",
    )   
    
    cat_1mu_6j_3bj = cat_6j_3bj.add_category(
        name="1mu_6j_3bj",
        id=cat_6j_3bj.id + cat_1mu.id,
        selection=[cat_1mu.selection, cat_6j_3bj.selection],
        label="1 $\mu$\n$\\geq 6$ jets\n$= 3$ b-jets",
    )  
    
    cat_1mu_5j_3bj = cat_5j_3bj.add_category(
        name="1mu_5j_3bj",
        id=cat_5j_3bj.id + cat_1mu.id,
        selection=[cat_1mu.selection, cat_5j_3bj.selection],
        label="1 $\mu$\n$= 5$ jets\n$= 3$ b-jets",
    )
    