#!/bin/bash

set -euo pipefail

echo "*** push submodules ***"

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
