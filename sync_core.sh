#!/bin/bash

set -euo pipefail

echo "*** syncing core ***"

if [ -z "$SSH_AGENT_PID" ]; then
	eval "$(ssh-agent -s)"
	ssh-add ~/.ssh/id_rsa
fi

cp 0_core/.gitattributes .

for d in [0-9]_*/ ; do

	if [ $d == "0_core/" ]; then
		continue
	fi

	if [ ! -f ${d}.git ] | [ ! -d ${d}.git ] ; then
		continue # .git is only a file in submodules
   fi

	(
	cd $d

	echo "* $d - syncinc core"

	if [ -n "$(git status -s)" ]; then
		echo "$d modified"
		exit 1
	fi

	rsync -au --exclude='.git' --exclude='requirements.txt' ../0_core/ .

	if [ "$(git status -s)" ]; then
		git add .
		git commit -m "SYNC CORE"
		git push
	fi
	)

done

while true; do
	read -p "save submodules? " yn
	case $yn in
		[Yy]* ) bash ./save_submodules.sh;  break;;
		[Nn]* ) exit;;
		* ) echo "Please answer yes or no.";;
	esac
done

echo "... done"
