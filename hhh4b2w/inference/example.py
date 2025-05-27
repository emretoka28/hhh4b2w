# coding: utf-8

"""
Example inference model.
"""

from columnflow.inference import inference_model, ParameterType, ParameterTransformation


@inference_model
def example(self):

    #
    # categories
    #

    self.add_category(
        "cat1",
        config_category="incl",
        config_variable="DNN_v10.output",
        #config_data_datasets=["data_mu_c"],
        mc_stats=True,
    )

    # self.add_category(
    #     "cat2",
    #     config_category="1e_5j_3bj",
    #     config_variable="DNN_v2.output",
    #     #config_data_datasets=["data_mu_c"],
    #     mc_stats=True,
    # )
    
    # self.add_category(
    #     "cat3",
    #     config_category="1mu_5j_3bj",
    #     config_variable="DNN_v2.output",
    #     #config_data_datasets=["data_mu_c"],
    #     mc_stats=True,
    # )
    # self.add_category(
    #     "cat2",
    #     config_category="2j",
    #     config_variable="jet1_eta",
    #     # fake data from TT
    #     data_from_processes=["TT"],
    #     mc_stats=True,
    # )

    # processes
    # HHH
    self.add_process(
        "c3_0_d4_0",
        is_signal=True,
        config_process="hhh_bbbbww_c3_0_d4_0",
        config_mc_datasets=["hhh_bbbbww_c3_0_d4_0_amcatnlo"],
    )
    # self.add_process(
    #     "c3_0_d4_99",
    #     is_signal=True,
    #     config_process="hhh_bbbbww_c3_0_d4_99",
    #     config_mc_datasets=["hhh_bbbbww_c3_0_d4_99_amcatnlo"],
    # )
    self.add_process(
        "c3_0_d4_m1",
        is_signal=True,
        config_process="hhh_bbbbww_c3_0_d4_m1",
        config_mc_datasets=["hhh_bbbbww_c3_0_d4_m1_amcatnlo"],
    )
    self.add_process(
        "c3_19_d4_19",
        is_signal=True,
        config_process="hhh_bbbbww_c3_19_d4_19",
        config_mc_datasets=["hhh_bbbbww_c3_19_d4_19_amcatnlo"],
    )
    self.add_process(
        "c3_m1_d4_0",
        is_signal=True,
        config_process="hhh_bbbbww_c3_m1_d4_0",
        config_mc_datasets=["hhh_bbbbww_c3_m1_d4_0_amcatnlo"],
    )
    self.add_process(
        "c3_m1_d4_m1",
        is_signal=True,
        config_process="hhh_bbbbww_c3_m1_d4_m1",
        config_mc_datasets=["hhh_bbbbww_c3_m1_d4_m1_amcatnlo"],
    )
    # self.add_process(
    #     "c3_1_d4_0",
    #     is_signal=True,
    #     config_process="hhh_bbbbww_c3_1_d4_0",
    #     config_mc_datasets=["hhh_bbbbww_c3_1_d4_0_amcatnlo"],
    # )
    # self.add_process(
    #     "c3_m1p5_d4_m0p5",
    #     is_signal=True,
    #     config_process="hhh_bbbbww_c3_m1p5_d4_m0p5",
    #     config_mc_datasets=["hhh_bbbbww_c3_m1p5_d4_m0p5_amcatnlo"],
    # )    
    # self.add_process(
    #     "c3_4_d4_9",
    #     is_signal=True,
    #     config_process="hhh_bbbbww_c3_4_d4_9",
    #     config_mc_datasets=["hhh_bbbbww_c3_4_d4_9_amcatnlo"],
    # )

    # Backgrounds
    self.add_process(
        "TT_SL",
        is_signal=False,
        config_process="tt_sl",
        config_mc_datasets=["tt_sl_powheg"],
    )
    self.add_process(
        "TT_DL",
        is_signal=False,
        config_process="tt_dl",
        config_mc_datasets=["tt_dl_powheg"],
    )
    self.add_process(
        "TT_FH",
        is_signal=False,
        config_process="tt_fh",
        config_mc_datasets=["tt_fh_powheg"],
    )
    self.add_process(
        "TTH",
        is_signal=False,
        config_process="tth",
        config_mc_datasets=["tth_hbb_powheg"],
    )
    self.add_process(
        "HH",
        is_signal=False,
        config_process="hh_ggf",
        config_mc_datasets=["hh_ggf_hbb_hvv_kl1_kt1_powheg"],
    )
    self.add_process(
        "TTHH",
        is_signal=False,
        config_process="ttHH",
        config_mc_datasets=["ttHHto4b_madgraph"],
    )


    #
    # parameters
    #

    # groups
    self.add_parameter_group("experiment")
    self.add_parameter_group("theory")

    # lumi
    lumi = self.config_inst.x.luminosity
    for unc_name in lumi.uncertainties:
        self.add_parameter(
            unc_name,
            type=ParameterType.rate_gauss,
            effect=lumi.get(names=unc_name, direction=("down", "up"), factor=True),
            transformations=[ParameterTransformation.symmetrize],
        )

    # # tune uncertainty
    # self.add_parameter(
    #     "tune",
    #     process="TT",
    #     type=ParameterType.shape,
    #     config_shift_source="tune",
    # )

    # # muon weight uncertainty
    # self.add_parameter(
    #     "mu",
    #     process=["ST", "TT"],
    #     type=ParameterType.shape,
    #     config_shift_source="mu",
    # )

    # # jet energy correction uncertainty
    # self.add_parameter(
    #     "jec",
    #     process=["ST", "TT"],
    #     type=ParameterType.shape,
    #     config_shift_source="jec",
    # )

    # # a custom asymmetric uncertainty that is converted from rate to shape
    # self.add_parameter(
    #     "QCDscale_ttbar",
    #     process="TT",
    #     type=ParameterType.shape,
    #     transformations=[ParameterTransformation.effect_from_rate],
    #     effect=(0.5, 1.1),
    # )


@inference_model
def example_no_shapes(self):
    # same initialization as "example" above
    example.init_func.__get__(self, self.__class__)()

    #
    # remove all shape parameters
    #

    for category_name, process_name, parameter in self.iter_parameters():
        if parameter.type.is_shape or any(trafo.from_shape for trafo in parameter.transformations):
            self.remove_parameter(parameter.name, process=process_name, category=category_name)
