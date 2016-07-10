#!/bin/bash
#
# use this to run a single script as a nimbix job
#
# use like:
#
# bash nimbix-run.sh gpuexperiments/occupancy_dyn.py
#
# pre-requisites:
# - nimbix account http://nimbix.net
# - must have installed https://github.com/hughperkins/nimbix-admin , and configured it
# - expects you to have an image called 'ngd3', with this repository installed to /data/git/gpu-experiments
#   and a script file at /data/git/gpu-experiments, which activates an appropriate virtualenv, and adds cuda
#   binaries to the PATH

scriptpath=$1

cat <<EOF>/tmp/run.sh
#!/bin/bash

cd /data/git/gpu-experiments
git pull
source ./setenv.sh
pip install -r requirements.txt
which python
python -V
python ${scriptpath}
EOF
cat /tmp/run.sh

nimbix-script --type ngd3 --image ngd3 /tmp/run.sh

