#!/bin/bash

set -euo pipefail

echo "*** syncing core ***"

if [ -z "$SSH_AGENT_PID" ]; then
	eval "$(ssh-agent -s)"
	ssh-add ~/.ssh/id_rsa
fi

cp 0_core/.gitattributes .

for d in [0-9]_*/ ; do

	if [ d == "0_core" ]; then
		continue
	fi

	(
	cd $d
	if [ ! $(git rev-parse --show-superproject-working-tree) ] ; then
		continue
	fi

	echo "* $d - syncinc core"

	if [ -n "$(git status -s)" ]; then
		echo "$d modified"
		continue
	fi

	rsync -au --exclude='.git' ../0_core/ .

	if [ -n "$(git status -s)" ]; then
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
