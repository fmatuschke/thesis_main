#!/bin/bash

set -euo pipefail

echo "*** save submodules ***"

if [ -z "$SSH_AGENT_PID" ]; then
	eval "$(ssh-agent -s)"
	ssh-add ~/.ssh/id_rsa
fi

for d in */ ; do

	if [ ! -f ${d}.git ] | [ ! -d ${d}.git ] ; then
		continue # .git is only a file in submodules
   fi

	(
	cd $d
	echo "* $d - save submodules"

	if [ -n "$(git status -s)" ]; then
		echo "found untracked files"
		git status
		while true; do
			read -p "commit? " yn
			case $yn in
				[Yy]* ) git add .; git commit -m "SAVE";  break;;
				[Nn]* ) exit;;
				* ) echo "Please answer yes or no.";;
			esac
		done

		if [ -n "$(git status -s)" ]; then
			echo "still modified"
			exit 1
		fi
	else
		echo "clean"
	fi
	)
done

while true; do
	read -p "push submodules? " yn
	case $yn in
		[Yy]* ) bash ./push_submodules.sh;  break;;
		[Nn]* ) exit;;
		* ) echo "Please answer yes or no.";;
	esac
done

echo "... done"
