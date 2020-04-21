#!/bin/bash

set -euo pipefail

echo "save submodules"

if [ -z "$SSH_AGENT_PID" ]; then
	eval "$(ssh-agent -s)"
	ssh-add ~/.ssh/id_rsa
fi

for d in */ ; do
	(
	cd $d
	if [ -z "$(git rev-parse --show-superproject-working-tree)" ] ; then
		# echo "$d no git submodule"
		continue
	fi

	if [ -n "$(git status -s)" ]; then
		echo "* $d found untracked files"
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
			echo "$d still modified"
			exit 1
		fi
	else
		echo "* $d clean"
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
