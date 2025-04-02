#!/usr/bin/env bash

# Function to prompt user for confirmation if command is supposed to be skipped
# if user types n or no then the command is executed
# if user types anything else or nothing then the command is skipped
# it is also possible to extend the command afte typing n or no

# Define color variables
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

confirm_and_run() {
    cmd="$1"
    echo -e "${YELLOW}$cmd${NC}"
    echo -e -n "    ${RED}Skip?${NC} (y/n): "
    read response
    case "$response" in
        [nN][oO]|[nN])
            echo -e "    ${RED}Any additional parameters?${NC} ('-ps' -> '--print-status 2,0', '-ro' -> '--remove-output 0,a,y' predefined, others possible)"
            read -e -i "    " extra_params

            # Add another case statement to handle the expansion
            case "$extra_params" in
                -ps)
                    extra_params="--print-status 2,0"
                    echo -e "    ${GREEN}Printing status...${NC}"
                    ;;
                -ro)
                    extra_params="--remove-output 0,a,y"
                    echo -e "    ${RED}Removing output...${NC}"
                    ;;
            esac
            echo -e "    ${GREEN}Running...${NC}"
            eval "$cmd $extra_params"
            ;;
        *)
            echo -e "    ${GREEN}Skipped!${NC}"
            ;;
    esac
}

# Define parameters
version=HHH_TT_FULL_weight_producer_empty
#HHH_TT_FULL
config=run3_2022_preEE_nano_v12
analysis=hhh4b2w.config.analysis_hhh4b2w.analysis_hhh4b2w
dataset=tt*,hhh_bbbbww_c3_0_d4_0_amcatnlo,hhh_bbbbww_c3_0_d4_99_amcatnlo,hhh_bbbbww_c3_0_d4_minus1_amcatnlo,hhh_bbbbww_c3_19_d4_19_amcatnlo,hhh_bbbbww_c3_1_d4_0_amcatnlo,hhh_bbbbww_c3_1_d4_2_amcatnlo,hhh_bbbbww_c3_2_d4_minus1_amcatnlo,hhh_bbbbww_c3_4_d4_9_amcatnlo,hhh_bbbbww_c3_minus1_d4_0_amcatnlo,hhh_bbbbww_c3_minus1_d4_minus1_amcatnlo
selector=example
mlmodel=DNN1
producer=ml_producer
calibrator=example
process=tt,hhh_bbbbww_c3_0_d4_0,hhh_bbbbww_c3_0_d4_99,hhh_bbbbww_c3_0_d4_minus1,hhh_bbbbww_c3_19_d4_19,hhh_bbbbww_c3_1_d4_0,hhh_bbbbww_c3_1_d4_2,hhh_bbbbww_c3_2_d4_minus1,hhh_bbbbww_c3_4_d4_9,hhh_bbbbww_c3_minus1_d4_0,hhh_bbbbww_c3_minus1_d4_minus1
categories=incl
variables=DNN1.output

# Define common arguments like version, config, analysis, workers
common_args=(
    "--version" "${version}"
    "--config" "${config}"
    "--analysis" "${analysis}"
    "--workers" "8"
)

# Define arguments for the ReduceEvents task 
# (triggers calibration, selection, and reduction of events)
args=(
    "--calibrator" "${calibrator}"
    "--selector" "${selector}"
    "--dataset" "${dataset}"
)

command_to_run="law run cf.ReduceEvents ${common_args[@]} ${args[@]}"
confirm_and_run "$command_to_run"

# Define arguments for the ProduceColumns tasks
# (here we produce columns for two different producers)
args1=(
    "--calibrator" "${calibrator}"
    "--producer" "${producer}"
    "--selector" "${selector}"
    "--dataset" "${dataset}"
)

args2=(
    "--calibrator" "${calibrator}"
    "--producer" "${producer}"
    "--selector" "${selector}"
    "--dataset" "${dataset}"
)

command_to_run="law run cf.ProduceColumns ${common_args[@]} ${args1[@]}"
confirm_and_run "$command_to_run"
command_to_run="law run cf.ProduceColumns ${common_args[@]} ${args2[@]}"

# Define arguments for the PlotVariables1D task
args=(
    "--calibrator" "${calibrator}"
    "--selector" "${selector}"
    "--producers" "${producer}"
   "--ml-models" "${mlmodel}"
    "--dataset" "${dataset}"
    "--processes" "${process}"
    "--categories" "${categories}"
    "--variables" "${variables}"
    "--plot-suffix" "norm"  # plot suffix to be added to the output file name
    "--hide-errors"  # don't show error bars in the plot
    "--skip-ratio"  # don't plot the MC/data ratio
    "--shape-norm"  # normalize the shapes to unity (useful for limited configs)
    "--cms-label" "pw"  # add the CMS label with the private work (pw) status
    "--file-types" "pdf"  # save the plot in PDF and PNG formats
    "--yscale" "linear"  # use linear scale for the y-axis
    "--process-setting" "tt,unstack,color=#832db6:hhh_bbbbww_c3_0_d4_0,unstack,color=#000000:hhh_bbbbww_c3_0_d4_99,unstack,color=#ffa90e:hhh_bbbbww_c3_0_d4_minus1,unstack,color=##bd1f01:hhh_bbbbww_c3_19_d4_19,unstack,color=#94a4a2:hhh_bbbbww_c3_1_d4_0,unstack,color=#3f90da:hhh_bbbbww_c3_1_d4_2,unstack,color=#a96b59:hhh_bbbbww_c3_2_d4_minus1,unstack,color=#e76300:hhh_bbbbww_c3_4_d4_9,unstack,color=#b9ac70:hhh_bbbbww_c3_minus1_d4_0,unstack,color=#717581:hhh_bbbbww_c3_minus1_d4_minus1,unstack,color=#92dadd"
    
)

command_to_run="law run cf.PlotVariables1D ${common_args[@]} ${args[@]}"
confirm_and_run "$command_to_run"
