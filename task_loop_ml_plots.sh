# Define parameters
version=ML_HHH_TT_FULL
config=run3_2022_preEE_nano_v12
analysis=hhh4b2w.config.analysis_hhh4b2w.analysis_hhh4b2w
hhh_dataset_names="hhh_bbbbww_c3_0_d4_0 hhh_bbbbww_c3_0_d4_99 hhh_bbbbww_c3_0_d4_minus1 hhh_bbbbww_c3_19_d4_19 hhh_bbbbww_c3_1_d4_0 hhh_bbbbww_c3_1_d4_2 hhh_bbbbww_c3_2_d4_minus1 hhh_bbbbww_c3_4_d4_9 hhh_bbbbww_c3_minus1_d4_0 hhh_bbbbww_c3_minus1_d4_minus1"
selector=example
producer=ml_producer
calibrator=example
mlmodel=PNN
process=tt,hhh_bbbbww_c3_0_d4_0,hhh_bbbbww_c3_0_d4_99,hhh_bbbbww_c3_0_d4_minus1,hhh_bbbbww_c3_19_d4_19,hhh_bbbbww_c3_1_d4_0,hhh_bbbbww_c3_1_d4_2,hhh_bbbbww_c3_2_d4_minus1,hhh_bbbbww_c3_4_d4_9,hhh_bbbbww_c3_minus1_d4_0,hhh_bbbbww_c3_minus1_d4_minus1
categories=incl,1e_6j_4bj,1e_5j_4bj,1e_6j_3bj,1mu_6j_4bj,1mu_5j_4bj,1mu_6j_3bj
variables=n_jet

args=(
    "--calibrator" "${calibrator}"
    "--selector" "${selector}"
    "--producers" "${producer}"
    "--ml-models" "${mlmodel}"
    "--categories" "${categories}"
    "--variables" "${variables}"
    "--plot-suffix" "norm"  # plot suffix to be added to the output file name
    "--hide-errors"  # don't show error bars in the plot
    "--skip-ratio"  # don't plot the MC/data ratio
    "--shape-norm"  # normalize the shapes to unity (useful for limited configs)
    "--cms-label" "pw"  # add the CMS label with the private work (pw) status
    "--file-types" "pdf"  # save the plot in PDF and PNG formats
    "--yscale" "linear"  # use linear scale for the y-axis
    "--process-setting" "hhh_bbbbww_c3_0_d4_0,unstack,color=#3f90da:tt,unstack,color=#2ca02c"
    #:hhh_bbbbww_c3_0_d4_99,unstack,color=ffa90e:hhh_bbbbww_c3_0_d4_minus1,unstack,color=##bd1f01:hhh_bbbbww_c3_19_d4_19,unstack,color=#94a4a2:hhh_bbbbww_c3_1_d4_0,unstack,color=#832db6:hhh_bbbbww_c3_1_d4_2,unstack,color=#a96b59:hhh_bbbbww_c3_2_d4_minus1,unstack,color=#e76300:hhh_bbbbww_c3_4_d4_9,unstack,color=#b9ac70:hhh_bbbbww_c3_minus1_d4_0,unstack,color=#717581:hhh_bbbbww_c3_minus1_d4_minus1,unstack,color=#92dadd

    #hhh_bbbbww_c3_0_d4_0
    #:hhh_bbbbww_c3_minus1_d4_minus1,unstack,color=#832db6:hhh_bbbbww_c3_19_d4_19,unstack,color=#e76300"
    #:tt,unstack,color=#94a4a2"
    #w_lnu,unstack,color=#a96b59:
)
common_args=(
    "--version" "${version}"
    "--config" "${config}"
    "--analysis" "${analysis}"
    "--workers" "4"
    # "--workflow" "htcondor"
    # "--htcondor-memory" "45"
)

task="cf.PlotVariables1D"

command_to_run="law run ${task} ${common_args[@]} ${args[@]}"

#starting with Plotting task 

for process in $hhh_dataset_names
do
    echo "####################### Processing ttbar + ${process} #######################"
    #echo "$command_to_run --dataset ${process}"
    eval "$command_to_run --datasets tt*,${process}* --processes tt,${process} --process-setting 'tt,unstack,color=#2ca02c:${process},unstack,color=#3f90da'"
    echo "####################### Done with ttbar + ${process} #######################"
    echo ""
done   
echo "####################### Done with all datasets. #######################"
