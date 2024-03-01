#!/bin/bash




#---------------------------------------------------------------------
#-------------- Some file information for easier  --------------------
#------------------- adaptation of pipeline --------------------------
#---------------------------------------------------------------------


ignore_scripts=(\
".git" \
".gitignore" \
)

configuration_files=(\
"config_variables.pickle" \
"check_variables.txt" \
"param_function.py" \
)

simulation_files=(\
".git" \
".gitignore" \
)

training_files=(\
".git" \
".gitignore" \
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
if [ $cluster == 1 ]
then

	#set -o errexit  # Exit the script on any error
	set -o nounset  # Treat any unset variables as an error

	module --quiet purge  # Reset the modules to the system default

	# Set the ${PS1} (needed in the source of the Anaconda environment)
	export PS1=\$

	module load Miniconda3/22.11.1-1

	source ${EBROOTMINICONDA3}/etc/profile.d/conda.sh

	conda deactivate &>/dev/null

	conda activate /fp/homes01/u01/ec-gertwk/.conda/envs/$swyft_env

	
else
	conda activate ${swyft_env}

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
make_new_results_dir=1
if [ -d $results_dir ]
then
	echo Found directory with same run name: ${results_dir}
	echo Sending output files there.
	make_new_results_dir=0
fi


# If results parent-folder doesn't exist, creates it.
if [ $make_new_results_dir == 1 ]
then
	echo -n "Making new results-directory with sub-directories... "
	mkdir ${results_dir}
	mkdir ${results_dir}/sim_output
	mkdir ${results_dir}/train_output
	mkdir ${results_dir}/val_output
	mkdir ${results_dir}/archive
	echo  done.
	echo
fi


#---------------------------------------------------------------------
#------ Copying files to results/$run_name_ext folder and ------------
#------------- archiving files from earlier trials -------------------
#---------------------------------------------------------------------

# Determining which run-number (i) is currently being executed, based
# on existence of files and folders ending in "__i".  
i=0
if find . -maxdepth 1 -type f -name '*__0.*' | grep -q 0 || [ -d ${results_dir}/archive/trial__0 ]; then i=1 ; fi
while find . -maxdepth 1 -type f -name '*__${i}.*' | grep -q 0 || [ -d ${results_dir}/archive/trial__$(($i - 1)) ]
do
	i=$(($i+1))
done
echo "Run number for this run name is ${i}."
echo

# Putting earlier files into archive folder (if i is greater than 0). 

echo -n "Making new archive_subfolder, and sending old files there... "
if [ $i == 0 ]; then mkdir ${results_dir}/archive/trial__0 ; fi
if [ $i -gt 0 ]
then	
	mkdir ${results_dir}/archive/trial__$(($i - 1))

	for file_name in $(find ${results_dir} -maxdepth 1 -type f )
	do
		mv $file_name ${results_dir}/archive/trial__$(($i - 1))			
	done
	
	if [ -d ${results_dir}/sim_output/store ] && [ $simulate == 1 ] && [ $sim_from_scratch == 1 ]; then rsync -r ${results_dir}/sim_output/store ${results_dir}/archive/trial__$(($i - 1)) ; fi
	if [ -d ${results_dir}/val_output/store ] && [ $simulate_val == 1 ] && [ $sim_val_from_scratch == 1 ]; then rsync -r ${results_dir}/val_output/store ${results_dir}/archive/trial__$(($i - 1)) ; fi		
fi
echo  done.


# Copying analysis_scripts into results folder. 

echo -n "Copying analysis scripts to results folder... "
cp $0 ${results_dir}
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

# Make possibility to copy store from somewhere else. (implement in python?)

if [ $update_config == 1 ] || [ $i == 0 ] ; then
	
	echo
	echo "Converting and saving new analysis variables"
	echo "------------------------------------------------------"
	echo
	
	python ${results_dir}/set_parameters.py -args "$1"
	
	echo -n "Writing new parameter-extension function... "
	python ${results_dir}/config_pois.py -results_dir $results_dir
	echo done.
	 
else
		
	echo "Checking for variables defined in previous runs... "
	for item in "${configuration_files[@]}"
	do	
		found=0
		check_i=$(($i - 1))
		while [ $found == 0 ] && [ $check_i -gt 0 ]
		do
			for file_name in $(find ${results_dir}/archive/trial__$check_i -maxdepth 1 -type f )
			do
				if [ "${results_dir}/archive/trial__$check_i/${item}" == "$file_name" ]; then
					found=1
					break
				fi
		    	done
		    	
			if [ $found == 1 ]; then 
				rsync $file_name ${results_dir}
				echo "Imported ${item} from run number $(($i - 1))"
			else
				check_i=$(($check_i - 1))  
			fi
		done
		if [ $found == 0 ]; then 
			echo "ERROR: ${item} was not found in previous runs!"
			exit 1
		fi		
	done
	echo
	
fi







#---------------------------------------------------------------------
#-------------------------- Simulation -------------------------------
#---------------------------------------------------------------------

# move old store to most recent existing archive folder. 
if [ $save_old_sims != 1 ] && [ $i != 0 ] ; then

	if [[ -e ${results_dir}/sim_output/store ]] ; then
		
		check_i=$(($i - 1))
		while [ -e ${results_dir}/archive/trial__(($check_i-1)) ] && [ $check_i -gt 0 ]
		do
			check_i=$(($check_i - 1))
		done
		
		if  [ $check_i > -1 ] ; then
			echo -n "Moving old store to archive/trial $(($check_i-1))... "
			rsync -r ${results_dir}/sim_output/store ${results_dir}/archive/trial__(($check_i-1))
			echo done. 
		else
			echo Error: No archive folders exist for previous runs. 
			exit 1
		fi
		
	fi
	
fi	

# Delete old store, unless chosen to keep, or if not saved. 
if [ $use_old_sims != 1 ] && [ $save_old_sims != 0 ] ; then

	# Delete existing store if it exists
	if [[ -e ${results_dir}/sim_output/store ]] ; then
		echo -n "Deleting old store... "
		rm -r ${results_dir}/sim_output/store
		echo done.
	fi
	
	# Make new store dir
	mkdir ${results_dir}/sim_output/store
	echo Made new empty store folder.
fi

# import store from elswhere, if so chosen
if [[ -e "$import_old_sims" ]] ; then

	# Copy store from elswhere
	rsync -r $import_old_sims/store ${results_dir}/sim_output/store
	rsync -r $import_old_sims/store ${results_dir}/sim_output/store.sync
	
	
if [ $import_old_sims == 1 ] ; then
	# Makes a new store folder if it doesn't already exist.
	if [[ ! -e ${results_dir}/sim_output/store ]] ; then
		mkdir ${results_dir}/sim_output/store
		echo Made new empty store folder.
	fi
fi



# Config simulation pipeline
# set store path (and initialize if necessary)
# simulate






# [ $simulate == 1 ]
#then
	
#	echo
#	echo "Simulation"
#	echo "------------------------------------------------------"
#	echo
	
#	${results_dir}/simulate.py
	
#	wait
	
#	mv sim_output/sim_output.txt sim_output/sim_output__${i}.txt
	

	







# Writing shell script and execution
# Make possibility to copy store from somewhere else. (implement in python?)


#---------------------------------------------------------------------
#---------------------------- Training -------------------------------
#---------------------------------------------------------------------


# Writing shell script and execution
# Make possibility to copy network from somewhere else. (implement in python?)
# Implement way to define which network architecture to use





#---------------------------------------------------------------------
#------------------------End of analysis -----------------------------
#---------------------------------------------------------------------

echo
echo Finished analysis. 
echo


exit

