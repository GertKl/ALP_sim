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
echo
echo "       = = = ========================== = = =          "
echo "                                                       "
echo "                      Starting                         "
echo "              ALP INFERENCE WITH NRE                   "
echo "                  w/ swyft 0.4.5                       "
echo "                                                       "
echo "                 Run name: ${run_name}                 "
echo "                                                       "
echo "       = = = ========================== = = =          "
echo "                                                       "
echo
echo


#---------------------------------------------------------------------
#-------------------Activating conda environment ---------------------
#---------------------------------------------------------------------

echo -n "Activating conda environment ${swyft_env}... "
if [[ $on_cluster == fox ]] || [[ $on_cluster == hepp ]]
then

	#set -o errexit  # Exit the script on any error
	set -o nounset  # Treat any unset variables as an error

	module --quiet purge  # Reset the modules to the system default

	# Set the ${PS1} (needed in the source of the Anaconda environment)
	export PS1=\$

	if [[ $on_cluster == fox ]] ; then	
		module load Miniconda3/22.11.1-1
	elif [[ $on_cluster == hepp ]] ; then
		module load Miniconda3/4.9.2
	fi

	source ${EBROOTMINICONDA3}/etc/profile.d/conda.sh

	conda deactivate &>/dev/null
fi

if [[ $on_cluster == fox ]] ; then
	conda activate /fp/homes01/u01/ec-gertwk/.conda/envs/$swyft_env
elif [[ $on_cluster == fox ]] || [[ $on_cluster == hepp ]] ; then
	conda activate ${swyft_env}
else
	echo ERROR: Cluster \"$on_cluster\" not recognized. 
	exit 1
fi
echo  done. 
echo

#---------------------------------------------------------------------
#----------- Establishing or identifying results folder --------------
#---------------------------------------------------------------------


# Checking if results parent-folder exists, otherwise creates it.  
if [ -d ${results_parent_dir} ] 
then
    echo "Confirmed directory: ${results_parent_dir}"
    echo
else
    mkdir ${results_parent_dir}
    echo "Created directory ${results_parent_dir}"
    echo
fi

# Checking if results folder exists in results parent-folder.
if [ -d $results_dir ]
then
	echo Found directory with same run name: ${results_dir}
	echo Sending output files there.
	i=1		# Index of run is 1 or higher (to be determined below) 
else
	echo -n "Making new results-directory with sub-directories... "
	mkdir ${results_dir}
	echo done.
	i=0		# Index of run. 
fi
echo

if [ ! -e ${results_dir}/sim_output ] ; then mkdir ${results_dir}/sim_output ; fi
if [ ! -e ${results_dir}/train_output ] ; then mkdir ${results_dir}/train_output ; fi
if [ ! -e ${results_dir}/val_output  ] ; then	mkdir ${results_dir}/val_output ; fi
if [ ! -e ${results_dir}/archive ] ; then mkdir ${results_dir}/archive ; fi



#---------------------------------------------------------------------
#------ Copying files to results/$run_name_ext folder and ------------
#------------- archiving files from earlier trials -------------------
#---------------------------------------------------------------------

# Determining which run-number (i) is currently being executed, based
# on existence of files and folders ending in "__i".  
echo -n "Checking for earlier runs with this name... "
#if [ ! -d ${results_dir}/archive/trial__0 ] ; then mkdir ${results_dir}/archive/trial__0 ; fi
#if find . -maxdepth 1 -type f -name '*__0.*' | grep -q 0 || [ -d ${results_dir}/archive/trial__0 ]; then i=1 ; fi
if [ $i != 0 ] ; then
	while find . -maxdepth 1 -type f -name '*__$(($i-1)).*' | grep -q 0 || [ -d ${results_dir}/archive/trial__$(($i-1)) ]
	do
		i=$(($i+1))
	done
fi
echo "this is run number ${i}."
echo

# Putting earlier files into archive folder (if i is greater than 0). 

#if [ $i == 0 ]; then mkdir ${results_dir}/archive/trial__0 ; fi
if [ $i -gt 0 ]
then	
	echo -n "Making new archive_subfolder, and sending old files there... "
	if [ ! -d ${results_dir}/archive/trial__$(($i - 1)) ] ; then mkdir ${results_dir}/archive/trial__$(($i - 1)) ; fi

	for file_name in $(find ${results_dir} -maxdepth 1 -type f )
	do
		mv $file_name ${results_dir}/archive/trial__$(($i - 1))			
	done
	echo  done.
	
fi



# Copying analysis_scripts into results folder. 

echo -n "Copying analysis scripts to results folder... "
cp $0 ${results_dir}
cp $config_file ${results_dir}
for file_name in $(find ${scripts_dir} -maxdepth 1 -type f )
do
	found=0
	for item in "${ignore_scripts[@]}"
	do
		if [ "${scripts_dir}/${item}" == "$file_name" ]; then
			found=1
			break
		fi
    	done
	if [ $found == 0 ]; then rsync $file_name ${results_dir} ; fi			
done
echo  done.
echo




#---------------------------------------------------------------------
#------------ Setting or updating analysis parameters ----------------
#---------------------------------------------------------------------


#if [ $update_config == 0 ] || [ $update_physics == 0 ] ; then	
echo "Checking for variables defined in previous runs... "
found_any=0
for item in "${configuration_files[@]}"
do	
	if [ ! -e $results_dir/$item ]; then
		found=0
		check_i=$(($i - 1))
		while [ $found == 0 ] && [ $check_i -gt -1 ] ; do
			for file_name in $(find ${results_dir}/archive/trial__$check_i -maxdepth 1 -type f )
			do
				if [ "${results_dir}/archive/trial__$check_i/${item}" == "$file_name" ]; then
					found=1
					break
				fi
		    	done
		    	
			if [ $found == 1 ]; then 
				found_any=1
				if [ $item == "physics_variables.pickle" ] && [ $save_physics == 0 ] ; then
					mv $file_name ${results_dir}
				else
					cp $file_name ${results_dir}
					echo "Copied ${item} from run number $(($i - 1))"
				fi
			else
				check_i=$(($check_i - 1))  
			fi
		done
		if [ $found == 0 ]; then 
			if [ $update_config == 0 ] && [ $i != 0 ] ; then
				echo "ERROR: ${item} was not found in previous runs!"
				echo
				exit 1
			fi
		fi
	fi		
done
if [ $found_any == 0 ] ; then echo "No previously defined variables found." ; fi
echo

	
#fi



if [ $update_config == 1 ] || [ $i == 0 ] ; then
	
	echo
	echo "Converting and saving new analysis variables"
	echo "------------------------------------------------------"
	echo
	
	if [ $update_config_on_cluster == 1 ] ; then
		echo -n Writing set_parameters.sh...
		python $results_dir/config_set_parameters.py -path $results_dir
		echo done.
		echo Submitting set_parameters.sh to cluster. Run squeue -u \"$USER\" to see status.
		echo Output for this step will be sent to $results_dir/set_parameters_output/slurm-\<job ID\>-$i.out.
		if [ ! -e $results_dir/set_parameters_output ] ; then mkdir $results_dir/set_parameters_output ; fi
		sbatch \
		--wait \
		--output=$results_dir/set_parameters_output/slurm-%j-$i.out \
		--job-name="set_parameters" \
		--account=$account \
		--partition=$partition_config \
		--time=$max_time_config \
		--mem-per-cpu=${max_memory_config}G \
		--qos=$qos_config \
		${results_dir}/set_parameters.sh "$1"
		echo Finished setting up parameters\!
	else
		python ${results_dir}/set_parameters.py -args "$1"
	fi
	
	#echo -n "Writing new parameter-extension function... "
	#python ${results_dir}/config_pois.py -results_dir $results_dir
	#echo done.
	#echo

fi





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
	if [[ -e ${results_dir}/sim_output/store ]] && [ $double_check_saved_sims == 1 ] ; then
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
	
	mv $results_dir/sim_output/sim_outputs $results_dir/sim_output/sim_outputs__$i

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
	if [[ -e ${results_dir}/train_output/net ]] && [ $double_check_saved_net == 1 ]; then
		echo -n "Deleting old net... "
		rm -r ${results_dir}/train_output/net
		echo done.
	fi
fi

# Making a new store directory if appropriate
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
	
	# echo "Running train.sh... "
	echo "Training in progress. Run squeue -u \"$USER\" to see status."
	if [[ $on_cluster == fox ]] ; then
		sbatch --wait $results_dir/train.sh
	elif [[ $on_cluster == local ]] || [[ $on_cluster == hepp ]] ; then
		$results_dir/train.sh
	else
		echo ERROR: Cluster \"$on_cluster\" not recognized. 
		exit 1
	fi
	
	mv $results_dir/train_output/train_outputs $results_dir/train_output/train_outputs__$i

	echo 
fi






#---------------------------------------------------------------------
#------------------------End of analysis -----------------------------
#---------------------------------------------------------------------

echo
echo Finished analysis. 
echo


exit

