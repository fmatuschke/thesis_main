#!/bin/bash

echo "syncing submodules"

if [ -z "$SSH_AGENT_PID" ]; then
	eval "$(ssh-agent -s)"
	ssh-add ~/.ssh/id_rsa
fi

git reset

for d in */ ; do

	(
	cd $d
	if [ -z $(git rev-parse --show-superproject-working-tree) ] ; then
		echo "$d no git repository"
		continue
	fi

	if [ -n "$(git status -s)" ]; then
		echo "$d modified"
		continue
	fi

   echo "$d"

	git push
	)

   git add $d

done

git commit -m "UPDATE SUBMODULES"
git push

echo "... done"
