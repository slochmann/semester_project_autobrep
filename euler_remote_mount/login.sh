#!/bin/bash
source .env
sshfs slochmann@euler.ethz.ch:/cluster/home/slochmann /home/sebi/MSc/3.Sem/semester_thesis/local-git/semester_project_autobrep/euler_remote/euler_project -o ssh_command='sshpass -p '$PASS' ssh' &
sshfs slochmann@euler.ethz.ch:/cluster/scratch/slochmann /home/sebi/MSc/3.Sem/semester_thesis/local-git/semester_project_autobrep/euler_remote/euler_scratch -o ssh_command='sshpass -p '$PASS' ssh' &
sshpass -p $PASS ssh  slochmann@euler.ethz.ch
