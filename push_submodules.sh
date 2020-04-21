#!/bin/bash

set -euo pipefail

echo "*** push submodules ***"

if [ -z "$SSH_AGENT_PID" ]; then
	eval "$(ssh-agent -s)"
	ssh-add ~/.ssh/id_rsa
fi

for d in */ ; do

	(
	cd $d
	if [ -z "$(git rev-parse --show-superproject-working-tree)" ] ; then
		continue
	fi

	echo "* $d - push submodules"

	if [ -n "$(git status -s)" ]; then
		echo "$d modified"
      exit 1
	fi

	git push
	)

   git add $d

done

echo "... done"
