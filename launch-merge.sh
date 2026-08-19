#!/bin/bash
set -e
export SHELL="$(which bash)"
cd /home/pino/humanify
if [ -f ~/.omp/agent/.env ]; then set -a; source ~/.omp/agent/.env; set +a; fi
exec omp --model freeinference/deepseek-v4-flash:high --auto-approve @task-merge.md
