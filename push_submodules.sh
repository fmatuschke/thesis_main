#!/bin/bash

set -euo pipefail

echo "syncing submodules"

if [ -z "$SSH_AGENT_PID" ]; then
	eval "$(ssh-agent -s)"
	ssh-add ~/.ssh/id_rsa
fi

# if [ -n "$(git status --ignore-submodules -s)" ] ; then
#    echo "git main repository not clean"
#    exit 1
# fi

for d in */ ; do

	(
	cd $d
	if [ -z "$(git rev-parse --show-superproject-working-tree)" ] ; then
		# echo "$d no git submodule"
		continue
	fi

	if [ -n "$(git status -s)" ]; then
		echo "$d modified"
      exit 1
	fi

   echo "* $d push"
	git push
	)

   git add $d

done

# git commit -m "UPDATE SUBMODULES"
# git push

echo "... done"
