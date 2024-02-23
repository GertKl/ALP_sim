#!/bin/bash

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

echo -n "Activiating conda environment ${swyft_env}"...
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
echo done. 
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
	echo
	make_new_results_dir=0
fi


# If results parent-folder doesn't exist, creates it.
if [ $make_new_results_dir == 1 ]
then
	echo -n Making new results-directory with sub-directories... 
	mkdir ${results_dir}
	mkdir ${results_dir}/sim_output
	mkdir ${results_dir}/train_output
	mkdir ${results_dir}/val_output
	mkdir ${results_dir}/archive
	echo done.
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
while find . -maxdepth 1 -type f -name '*__${i}.*' | grep -q 0 || [ -d ${results_dir}archive/trial__$(($i - 1)) ]
do
	i=$(($i+1))
done


# Copying this file into results folder, and putting earlier files
# into archive folder (if i is greater than 0). 

if [ $i == 0 ]; then mkdir mkdir ${results_dir}/archive/trial__0

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

cp $0 ${results_dir}

# Copying analysis_scripts into results folder. 


ignore_scripts=(".git" ".gitignore")


for file_name in $(find ${scripts_dir} -maxdepth 1 -type f )
do
	found=0
	for item in "${ignore_scripts[@]}"
	do
		if [ "$item" == "$file_name" ]; then
			found=1
			break
		fi
    	done
	if [ $found == 0 ]; then cp $file_name ${results_dir} ; fi			
done







#---------------------------------------------------------------------
#------------------------End of analysis -----------------------------
#---------------------------------------------------------------------

echo
echo Finished analysis. 
echo


exit