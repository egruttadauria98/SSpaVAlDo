#!/bin/bash

set -e

sudo /etc/init.d/ssh start &

/usr/local/bin/start.sh "$@"
