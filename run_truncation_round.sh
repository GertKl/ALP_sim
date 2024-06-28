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



#---------------------------------------------------------------------
#------------------------- Printing a banner -------------------------
#---------------------------------------------------------------------



echo
echo "Starting truncation round $2                          "
echo "= = = = = = = = = = = = = = = = = = = = = = = = = = = "
echo "                                                      "


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

double_check_saved_sims=1
# move old store to most recent existing archive folder. 
if [ $save_old_sims == 1 ] && [ $i != 0 ] ; then
	old_store_files=()
	double_check_saved_sims=0
	for file_name in $(find $results_dir/sim_output/store) ; do
		old_store_files+=("$file_name")
	done
	if [[ -e ${results_dir}/sim_output/store ]] ; then	
		
		#Redundant loop; remove?
		check_i=$(($i - 1))
		while [ ! -e ${results_dir}/archive/trial__$(($check_i-1)) ] && [ $check_i -gt 0 ] ; do
			check_i=$(($check_i-1))
		done
		
		if  [ $check_i > -1 ] && [[ -e ${results_dir}/sim_output/store ]] ; then
			echo -n "Copying old store to archive/trial__$(($check_i))... "
			rsync -r ${results_dir}/sim_output/store ${results_dir}/archive/trial__$(($check_i))
			echo done.
			if [ -e ${results_dir}/archive/trial__$(($check_i))/store ] ; then
				double_check_saved_sims=1
				for store_file in ${old_store_files[@]} ; do
					remove_string=${results_dir}/sim_output/store
					if [ ! -e ${results_dir}/archive/trial__$(($check_i))/store/${store_file/${remove_string}} ]
					then
						double_check_saved_sims=0
						echo ERROR: File ${results_dir}/archive/trial__$(($check_i))/store/${store_file/${remove_string}} was not copied succesfully from old store!
						exit 1
					fi
				done
			fi
		else
			if [ $check_i < 0 ] ; then
				echo Error: No archive folders exist for previous runs. 
				echo
				exit 1
			elif [[ -e ${results_dir}/sim_output/store ]] ; then
				echo No old store found. 
			fi
		fi	
	fi	
fi	

# Checking if $use_old_sims is specified appropriately 
if [ $use_old_sims != 1 ]  && [ $use_old_sims != 0 ] && [[ ! -e $use_old_sims  ]]; then
	echo Error: use_old_sims is invalid. Must be 1, 0, or an existing path. 
	echo
	exit 1
fi

# Delete old store, unless chosen to keep, or if not saved. 
if [ -e $use_old_sims ] \
|| { [ $use_old_sims == 0 ] && [ $simulate == 1 ]; } \
; then
	# Delete existing store if it exists
	if [[ -e ${results_dir}/sim_output/store ]] && [ $double_check_saved_sims == 1 ] && [ $2 == 0 ]; then
		echo -n "Deleting old store... "
		rm -r ${results_dir}/sim_output/store
		echo done.
	fi
fi

# Making a new store directory if appropriate
if [[ ! -e ${results_dir}/sim_output/store ]] ; then
	# Make new store dir
	mkdir ${results_dir}/sim_output/store
	echo Made new empty store folder in sim_output.
	echo
fi

# import store from elswhere, if so chosen
if [[ -e "$use_old_sims" ]] ; then
	echo -n "Importing the store $use_old_sims(.sync)... "
	rsync -r $use_old_sims ${results_dir}/sim_output/store
	rsync -r $use_old_sims.sync ${results_dir}/sim_output/store
	rsync $use_old_sims.lock.file ${results_dir}/sim_output/store
	echo done.
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


# ************Move network architecture file here***********************


# move old net to most recent existing archive folder. 
double_check_saved_net=1
if [ $save_old_net == 1 ] && [ $i != 0 ] ; then
	old_net_files=()
	double_check_saved_net=0
	for file_name in $(find $results_dir/train_output/net) ; do
		old_net_files+=("$file_name")
	done
	if [[ -e ${results_dir}/train_output/net ]] ; then	
		
		#Redundant loop; remove?
		check_i=$(($i - 1))
		while [ ! -e ${results_dir}/archive/trial__$(($check_i-1)) ] && [ $check_i -gt 0 ] ; do
			check_i=$(($check_i-1))
		done
		
		if  [ $check_i > -1 ] && [[ -e ${results_dir}/train_output/net ]] ; then
			echo -n "Copying old net to archive/trial__$(($check_i))... "
			rsync -r ${results_dir}/train_output/net ${results_dir}/archive/trial__$(($check_i))
			echo done. 
			if [ -e ${results_dir}/archive/trial__$(($check_i))/net ] ; then
				double_check_saved_net=1
				for net_file in ${old_net_files[@]} ; do
					remove_string=${results_dir}/train_output/net
					if [ ! -e ${results_dir}/archive/trial__$(($check_i))/net/${net_file/${remove_string}} ]
					then
						double_check_saved_net=0
						echo ERROR: File ${results_dir}/archive/trial__$(($check_i))/net/${net_file/${remove_string}} was not copied succesfully from old net!
						exit 1
					fi
				done
			fi
		else
			if [ $check_i < 0 ] ; then
				echo Error: No archive folders exist for previous runs. 
				echo
				exit 1
			elif [[ -e ${results_dir}/train_output/net ]] ; then
				echo No old net found. 
			fi
		fi	
	fi	
fi	





# Checking if $use_old_sims is specified appropriately 
if [ $use_old_net != 1 ]  && [ $use_old_net != 0 ] && [[ ! -e $use_old_net  ]]; then
	echo Error: use_old_net is invalid. Must be 1, 0, or an existing path. 
	echo
	exit 1
fi

# Delete old net, unless chosen to keep, or if not saved. 
if [ -e $use_old_net ] \
|| { [ $use_old_net == 0 ] && [ $train == 1 ]; } \
; then
	# Delete existing store if it exists
	if [[ -e ${results_dir}/train_output/net ]] && [ $double_check_saved_net == 1 ] && [ $2 == 0 ]; then
		echo -n "Deleting old net... "
		rm -r ${results_dir}/train_output/net
		echo done.
	fi
fi

# Making a new net directory if appropriate
if [[ ! -e ${results_dir}/train_output/net ]] ; then
	# Make new net dir
	mkdir ${results_dir}/train_output/net
	echo Made new empty net folder in train_output.
	echo
fi

# import net from elswhere, if so chosen
if [[ -e "$use_old_net" ]] ; then
	echo -n "Importing the net $use_old_net... "
	rsync -r $use_old_net ${results_dir}/train_output/net
	echo done.
	echo
fi

# import architecture from elswhere, if so chosen
if [[ -e "$architecture" ]] && [ $train == 1 ]  ; then
	echo -n "Importing the architecture $architecture... "
	rsync $architecture ${results_dir}/train_output/net/"network.py"
	echo done.
	echo
elif [[ "$architecture" == "" ]] && [ $train == 1 ] ; then
	echo -n "Importing the architecture $scripts_dir/network.py... "
	rsync -r $scripts_dir/network.py ${results_dir}/train_output/net/
	echo done.
	echo
elif [ $train == 1 ] ; then
	echo Error: Architecture badly defined. Should be an existing python file, or \"\". 
	echo
	exit 1
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





if [ $draw_DRP == 1 ] ; then
	
	# Drawing DRP samples
	echo "Drawing DRP coverage samples... "
	python $results_dir/draw_DRP_samples.py -path $results_dir
	echo "Done drawing DRP coverage samples."
	echo 

fi





exit

