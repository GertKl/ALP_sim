#!/bin/bash





#---------------------------------------------------------------------
#-------------- Some file information for easier  --------------------
#------------------- adaptation of pipeline --------------------------
#---------------------------------------------------------------------


ignore_scripts=(\
".git" \
".gitignore" \
"network.py" \
)

configuration_files=(\
"config_variables.pickle" \
"physics_variables.pickle" \
"check_variables.txt" \
"param_function.py" \
)

stopping_states=(\
"FAILED" \
"CANCELLED" \
"CANCELLED+" \
"COMPLETED" \
"TIMEOUT" \
"PREEMPTED" \
"NODE_FAIL" \
"OUT_OF_MEMORY" \
"OUT_OF_ME+" \
)

running_states=(\
"PD" \
"PENDING" \
"RUNNING" \
"SUSPENDED" \
)



#---------------------------------------------------------------------
#------------Declaring all variables from config-file ----------------
#---------------------------------------------------------------------


IFS=';' read -ra config_arguments <<< $1
for config_argument in "${config_arguments[@]}" 
do
	IFS='=' read -ra config_argument_name_and_value <<< $config_argument
	declare "${config_argument_name_and_value[0]}"="${config_argument_name_and_value[1]// /}"
done
echo

i=$4

#---------------------------------------------------------------------
#------------------------- Printing a banner -------------------------
#---------------------------------------------------------------------



echo
echo "Starting truncation round $2 (grid point $3)            "
echo "= = = = = = = = = = = = = = = = = = = = = = = = = = = = "
echo 


#---------------------------------------------------------------------
#-------------------------- Simulation -------------------------------
#---------------------------------------------------------------------

#Printing banner
if [ $simulate == 1 ] \
|| [ $save_old_sims == 1 ] \
|| [[ -e $use_old_sims ]] \
|| { [ $use_old_sims == 0 ] && [ $simulate == 1 ]; } \
; then
	echo
	echo "Simulation"
	echo "------------------------------------------------------"
	echo

fi




if [ $simulate == 1 ] ; then

	# Configure simulation pipeline
	echo "Writing simulate.sh... "
	python $results_dir/config_simulate.py -path $results_dir
	chmod +x $results_dir/simulate.sh
	echo "Done writing simulate.sh."
	echo 
	
	# Run simulate.sh
	
	list1_str=$(IFS=, ; echo "${running_states[*]}")
	list2_str=$(IFS=, ; echo "${stopping_states[*]}")
	
	echo "Running simulate.sh... "
	$results_dir/simulate.sh ${list1_str} ${list2_str}
	
	#mv $results_dir/sim_output/sim_outputs $results_dir/sim_output/sim_outputs__$i

	echo 
fi




#---------------------------------------------------------------------
#---------------------------- Training -------------------------------
#---------------------------------------------------------------------




#Printing banner
if [ $train == 1 ] \
|| [ $save_old_net == 1 ] \
|| [[ -e $use_old_net ]] \
|| { [ $use_old_net == 0 ] && [ $train == 1 ]; } \
; then
	echo
	echo "Training"
	echo "------------------------------------------------------"
	echo

fi



	

if [ $train == 1 ] ; then
	
	# Configure training pipeline
	echo "Writing train.sh... "
	python $results_dir/config_train.py -path $results_dir
	chmod +x $results_dir/train.sh
	echo "Done writing train.sh."
	echo 
	
	# Run simulate.sh
	
	echo "Running train.sh... "
	if [[ $on_cluster == fox ]] ; then
		echo "Training in progress. Run squeue -u \"$USER\" to see status."
		sbatch --wait $results_dir/train.sh
	elif [[ $on_cluster == local ]] || [[ $on_cluster == hepp ]] ; then
		$results_dir/train.sh
	else
		echo ERROR: Cluster \"$on_cluster\" not recognized. 
		exit 1
	fi
	
	#mv $results_dir/train_output/train_outputs $results_dir/train_output/train_outputs__$i

	echo 
fi








exit

