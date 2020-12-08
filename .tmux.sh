#!/bin/bash

if ! type "tmux" > /dev/null; then
	# echo "tmux not found"
	exit 1
fi

if [ $# -eq 0 ]; then
	SESSIONNAME=${PWD##*/}
else
	SESSIONNAME=$1
fi

tmux has-session -t $SESSIONNAME 2> /dev/null

if [ $? != 0 ] ; then
	tmux new-session -s $SESSIONNAME -n script -d
	tmux send-keys 'env-activate' Enter
	tmux send-keys 'clear' Enter

	# create layout array
	#tmux split-window -h
	tmux split-window -v

	# configure panes
	tmux select-pane -t 1
	tmux resize-pane -t 1 -y 5
	tmux send-keys 'htop -s PERCENT_CPU' Enter


	tmux select-pane -t 0
fi

TERM=xterm-256color tmux attach -t $SESSIONNAME
