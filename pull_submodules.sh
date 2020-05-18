#!/bin/bash

set -euo pipefail

echo "*** pull submodules ***"

# git submodule update --init --recursive

for d in */ ; do

	if [ ! -f ${d}.git ] | [ ! -d ${d}.git ] ; then
		continue # .git is only a file in submodules
   fi

	(
	cd $d

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
