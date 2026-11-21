#!/bin/bash 
#SBATCH -J 560anneal
#SBATCH -p xhacnormalc
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --time 72:00:00
#SBATCH --mem=122624M
#SBATCH --comment={appType:LAMMPS,appVersion:stable_29Aug2024,appVersionName:stable_29Aug2024,appShopId:,submitWay:portal}
#SBATCH -o std.out.%j
#SBATCH -e std.err.%j


#########################################################
### Slurm scripts for Sugon Portal_5.0 of LAMMPS   ### slurm2pbs  
### Version 1.0    |  2020-10-27  |  created by Yao Lang
#########################################################

### Get parameters from GUI

MIDFILE_DIR=/work/home/accnkmt08b/use/.portal
source $MIDFILE_DIR/job_portal.var
source $MIDFILE_DIR/job_interface.var

### Set basic var   ### MARK_slurm2pbs

JOBID=$SLURM_JOB_ID                                  ### slurm2pbs
QUEUE=$SLURM_JOB_PARTITION                           ### slurm2pbs 
NP=$SLURM_NPROCS                                     ### slurm2pbs
NNODE=`srun hostname | sort -V | uniq | wc -l`          ### slurm2pbs

INPUT_FILE=$WORK_DIR/job_${JOB_NAME}_${JOBID}.inp
OUTPUT_FILE=$WORK_DIR/job_${JOB_NAME}_${JOBID}.out
LOG_FILE=$WORK_DIR/job_${JOB_NAME}_${JOBID}.log
HOST_FILE=$WORK_DIR/job_${JOB_NAME}_${JOBID}_${NP}c_${NNODE}n.hosts

HOST_STRING=`srun hostname | sort -V | uniq`
echo $HOST_STRING > $HOST_FILE                       ### slurm2pbs

### Write basic job infomations

echo -e "The start time is: `date +"%Y-%m-%d %H:%M:%S"` \n" | tee -a $LOG_FILE 
echo -e "My job ID is: $JOBID \n" | tee -a $LOG_FILE  
echo -e "The total cores is: $NP \n" | tee -a $LOG_FILE 
echo -e "The hosts is: \n" | tee -a $LOG_FILE
cat $HOST_FILE | tee -a $LOG_FILE
echo -e "\n"  | tee -a $LOG_FILE 

### Confirm input file

if [ "x$IN_INPUT_FILE" == "x" ];then
 echo "You must specify a .in lammps script file." 1>&2
 exit 1
fi
Filename=`basename "$IN_INPUT_FILE"`
if [ $(dirname "$IN_INPUT_FILE") != $WORK_DIR ]; then
    cp $IN_INPUT_FILE $WORK_DIR
fi
array=(`echo $INCLUDE_FILE | tr ':' ' '` )
for var in ${array[@]}
do
  if [ -n "$var" ] && [ $(dirname "$var") != $WORK_DIR ]; then
    cp -rf "$var" "$WORK_DIR" 2>/dev/null
  fi
done
if [ ! -z $DATA_FILE ];then
    if [ $(dirname "$DATA_FILE") != $WORK_DIR ]; then
        cp $DATA_FILE $WORK_DIR
    fi
fi

### Run LAMMPS

if [[ -z $NGPU ]] || [[ $NGPU -eq 0 ]];then OPT_GPU="" ; fi
if [[ -n $NGPU ]] && [[ $NGPU -ge 1 ]];then OPT_GPU="-sf gpu -pk gpu $NGPU" ; fi

cd $WORK_DIR
#module load compiler/intel/intel-compiler-2017.5.239 mpi/intelmpi/2017.4.239
module purge

echo $MODULE_ENV > $MIDFILE_DIR/app.env
sed -i 's/_USER_COLON_SIGN_IN_GV_/:/g' $MIDFILE_DIR/app.env
source $MIDFILE_DIR/app.env

mpirun $APP $OPT_GPU -in $WORK_DIR/$Filename 2>&1
if [ $? -ne 0 ]; then
  exit 1
fi
echo The end time is: `date +"%Y-%m-%d %H:%M:%S"` | tee -a $LOG_FILE  
 
