#!/bin/bash

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR_TO_CLEAN="$PROJECT_ROOT/data/saved_index"
LOG_FILE="$PROJECT_ROOT/scripts/cleanup.log"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

log_message "Starting cleanup service..."
log_message "DIR_TO_CLEAN: $DIR_TO_CLEAN"
log_message "LOG_FILE: $LOG_FILE"

while true; do
    # Check if directory exists
    log_message "Checking if directory exists"

    if [ ! -d "$DIR_TO_CLEAN" ]; then
        log_message "ERROR: Directory $DIR_TO_CLEAN does not exist"
        # sleep for 24 hours
        continue
    fi

    # Remove contents
    rm -rf "$DIR_TO_CLEAN"/*
    log_message "Successfully cleared contents of $DIR_TO_CLEAN at $(date '+%Y-%m-%d %H:%M:%S')"
    
    # Wait for 24 hours
    sleep 86400
done
