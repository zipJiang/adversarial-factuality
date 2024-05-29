#!/bin/bash

# bind port 9871 to brtx602:9871
LOCAL_PORT=9871
REMOTE_HOST=localhost
REMOTE_PORT=9871
SSH_USER=zjiang31
SSH_SERVER=brtx602

if lsof -i :${LOCAL_PORT} | grep LISTEN; then
    echo "Port ${LOCAL_PORT} is already in use"
else
    echo "Binding port ${LOCAL_PORT} to ${SSH_SERVER}:${REMOTE_PORT}"
    ssh -o StrictHostKeyChecking=no -N -f -L ${LOCAL_PORT}:${REMOTE_HOST}:${REMOTE_PORT} ${SSH_USER}@${SSH_SERVER}

    if [ $? -eq 0 ]; then
        echo "Port binding successful."
    else
        echo "Port binding failed."
    fi
fi