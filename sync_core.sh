#!/bin/bash

set -e

echo "syncing core"

if [ -z "$SSH_AGENT_PID" ]; then
	eval "$(ssh-agent -s)"
	ssh-add ~/.ssh/id_rsa
fi

for d in [0-9]_*/ ; do

	if [ d == "0_core" ]; then
		continue
	fi

	(
	echo "$d"
	cd $d
	if [ ! $(git rev-parse --show-superproject-working-tree) ] ; then
		echo "$d no git repository"
		continue
	fi

	if [ -n "$(git status -s)" ]; then
		echo "$d modified"
		continue
	fi

	rsync -au --exclude='.git' ../0_core/ .

	if [ -n "$(git status -s)" ]; then
		git add .

		git commit -m "SYNC CORE"
		git push
		echo ""
	fi
	)

done

echo "... done"
