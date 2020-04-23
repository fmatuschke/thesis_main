#!/bin/bash

set -euo pipefail

echo "*** pull submodules ***"

git submodule update --init --recursive

for d in */ ; do

	(
	cd $d
	if [ -z "$(git rev-parse --show-superproject-working-tree)" ] ; then
		continue
	fi

	echo "* $d - pull submodules"

	if [ -n "$(git status -s)" ]; then
		echo "$d modified"
      exit 1
	fi

	if [ `git branch --list development` ] ; then
		git checkout development &> /dev/null
	else
		git checkout master &> /dev/null
	fi
   git pull
	)

done
