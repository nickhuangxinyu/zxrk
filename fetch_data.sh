#!/bin/bash
set -e
mkdir data
rsync -avz -r -e "sshpass -p wq101 ssh" wq101@139.196.204.22:~/tick* data/
rsync -avz -r -e "sshpass -p wq101 ssh" wq101@139.196.204.22:~/trans* data/
xz -d data/*
cat trans.20200423 | grep 0 > trans
