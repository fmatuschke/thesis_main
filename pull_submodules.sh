#!/bin/bash

set -euo pipefail

git submodule update --init --recursive

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

   echo "* $d pull"
   git checkout master
   git pull
	)

done
