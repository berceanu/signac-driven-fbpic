#!/bin/bash

PR_DIR=$(sed -n '2p' signac.rc | cut -d "=" -f2 | xargs)
SOURCE=${PR_DIR%/}
echo "${SOURCE}"

BACKUP_DIR="$(dirname "${SOURCE}")"
BACKUP_FILE="$(basename "${SOURCE}")"
DATE=$(date +%Y-%m-%d-%H%M%S)
echo "${BACKUP_DIR}/${BACKUP_FILE}-${DATE}.tar.gz"

tar -cvzpf ${BACKUP_DIR}/${BACKUP_FILE}-${DATE}.tar.gz ${SOURCE} > /dev/null 2>&1 &
