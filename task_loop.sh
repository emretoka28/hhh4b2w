# Define parameters
version=HHH_TT_FULL
versionML=HHH_TT_FULL
config=run3_2022_preEE_nano_v12
analysis=hhh4b2w.config.analysis_hhh4b2w.analysis_hhh4b2w
datasets="tt_sl_powheg tt_dl_powheg tt_fh_powheg hhh_bbbbww_c3_0_d4_0_amcatnlo hhh_bbbbww_c3_0_d4_99_amcatnlo hhh_bbbbww_c3_0_d4_minus1_amcatnlo hhh_bbbbww_c3_19_d4_19_amcatnlo hhh_bbbbww_c3_1_d4_0_amcatnlo hhh_bbbbww_c3_1_d4_2_amcatnlo hhh_bbbbww_c3_2_d4_minus1_amcatnlo hhh_bbbbww_c3_4_d4_9_amcatnlo hhh_bbbbww_c3_minus1_d4_0_amcatnlo hhh_bbbbww_c3_minus1_d4_minus1_amcatnlo"
ml_datasets="tt_sl_powheg tt_dl_powheg tt_fh_powheg hhh_bbbbww_c3_0_d4_0_amcatnlo hhh_bbbbww_c3_0_d4_99_amcatnlo hhh_bbbbww_c3_0_d4_minus1_amcatnlo hhh_bbbbww_c3_19_d4_19_amcatnlo hhh_bbbbww_c3_1_d4_0_amcatnlo hhh_bbbbww_c3_1_d4_2_amcatnlo hhh_bbbbww_c3_2_d4_minus1_amcatnlo hhh_bbbbww_c3_4_d4_9_amcatnlo hhh_bbbbww_c3_minus1_d4_0_amcatnlo hhh_bbbbww_c3_minus1_d4_minus1_amcatnlo"
selector=example
producer=example
producer2=ml_producer
calibrator=example


args1=(
        "--calibrator" "${calibrator}"
        "--producer" "${producer}"
        "--selector" "${selector}"
        "--version" "${version}"
    ) 

args2=(
        "--calibrator" "${calibrator}"
        "--selector" "${selector}"
        "--version" "${version}"
    ) 

args3=(
        "--calibrator" "${calibrator}"
        "--producer" "${producer2}"
        "--selector" "${selector}"
        "--version" "${versionML}"
    )

common_args=(
    "--config" "${config}"
    "--analysis" "${analysis}"
    "--workers" "4"
    "--workflow" "htcondor"
    "--htcondor-memory" "45"
)

task1="cf.ProduceColumns"
task2="cf.ReduceEvents"

command_to_run1="law run ${task1} ${common_args[@]} ${args1[@]}"
command_to_run2="law run ${task2} ${common_args[@]} ${args2[@]}"
command_to_run3="law run ${task1} ${common_args[@]} ${args3[@]}"


if [[ "$1" == "Produce" ]]; then
    echo "Running 'ProduceColumns' task over all provided datasets..."
    command_to_run=$command_to_run1
elif [[ "$1" == "Reduce" ]]; then
    echo "Running 'ReduceEvents' task over all provided datasets..."
    command_to_run=$command_to_run2
elif [[ "$1" == "ProduceML" ]]; then
    echo "Running 'ProduceColumns' task with ML-Producer over all provided datasets..."
    command_to_run=$command_to_run3    
else
    echo "Invalid argument. Please provide either 'Produce', 'Reduce' or 'ProduceML' as an argument."
    exit 1
fi

#starting with Reduce/Produce task loop

for dataset in $datasets
do
    echo "####################### Processing dataset: ${dataset} #######################"
    echo "$command_to_run --dataset ${dataset}"
    eval "$command_to_run --dataset ${dataset}"
    echo "####################### Done with dataset: ${dataset} #######################"
    echo ""
done   
    echo "####################### Done with all datasets. #######################"
