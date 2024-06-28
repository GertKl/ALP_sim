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
elif [[ $on_cluster == local ]] || [[ $on_cluster == hepp ]] ; then
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
#---------- Redirecting output to stdout and output.log   ------------
#---------------------------------------------------------------------


LOG_FILE=${results_dir}/output.log

exec > >(tee -a "$LOG_FILE") 2>&1





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
#---------------------- Truncation rounds ----------------------------
#---------------------------------------------------------------------



python ${results_dir}/run_truncation_rounds.py -argsstr "$1" -path "$results_dir" 


if [ $simulate == 1 ] ; then
	mv $results_dir/sim_output/sim_outputs $results_dir/sim_output/sim_outputs__$i
fi

if [ $train == 1 ] ||  [ $draw_DRP == 1 ] ; then
	mv $results_dir/train_output/train_outputs $results_dir/train_output/train_outputs__$i
fi







#---------------------------------------------------------------------
#------------------------End of analysis -----------------------------
#---------------------------------------------------------------------

echo
echo Finished analysis. 
echo




exit

