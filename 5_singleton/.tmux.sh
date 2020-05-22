SESSIONNAME=${PWD##*/}
tmux has-session -t $SESSIONNAME 2> /dev/null

if [ $? != 0 ]
 then
    tmux new-session -s $SESSIONNAME -n script -d

    # create layout array
    tmux split-window -h
    tmux split-window -v

    # configure panes
    tmux select-pane -t 1
    tmux send-keys 'htop -s PERCENT_CPU' C-m

    tmux select-pane -t 0
fi

TERM=xterm-256color tmux attach -t $SESSIONNAME
