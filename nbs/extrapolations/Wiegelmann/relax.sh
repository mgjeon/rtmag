#!/bin/bash

start_timestamp=$(date "+%Y-%m-%d %H:%M:%S.%3N")

# Function to calculate elapsed time and add timestamps to log
calculate_elapsed_time() {
    end_timestamp=$(date "+%Y-%m-%d %H:%M:%S.%3N")
    echo "" >> "$log_file"
    echo "Start timestamp: $start_timestamp" >> "$log_file"
    echo "End timestamp: $end_timestamp" >> "$log_file"
    
    start_time_seconds=$(date -d "$start_timestamp" +%s%3N)
    end_time_seconds=$(date -d "$end_timestamp" +%s%3N)
    elapsed_milliseconds=$((end_time_seconds - start_time_seconds))
    elapsed_seconds=$((elapsed_milliseconds / 1000))
    elapsed_milliseconds=$((elapsed_milliseconds % 1000))
    elapsed_time=$(printf "%02d:%02d:%02d.%03d" $(($elapsed_seconds/3600)) $(($elapsed_seconds%3600/60)) $(($elapsed_seconds%60)) $elapsed_milliseconds)
    echo "Elapsed time: $elapsed_time" >> "$log_file"
}

# Check if at least one argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <argument>"
    exit 1
fi

# Access the first argument
argument1=$1
argument2=$2

trap calculate_elapsed_time EXIT

if [ $argument1 == 23 ]; then
    log_file="relax_23.log"
    echo "Calculate potential field only (B0.bin)"
    echo "max. number of iterations: $argument2"
    ./relax1 $argument1 $argument2
    echo "Script execution completed."
elif [ $argument1 == 20 ]; then
    log_file="relax_20.log"
    echo "Calculate NLFFF only (Bout.bin)"
    echo "max. number of iterations: $argument2"
    ./relax1 $argument1 $argument2
    echo "Script execution completed."
elif [ $argument1 == 22 ]; then
    log_file="relax_22.log"
    echo "Calculate both potential & NLFFF (B0.bin & Bout.bin)"
    echo "max. number of iterations: $argument2"
    ./relax1 $argument1 $argument2
    echo "Script execution completed."
else
    echo "Invalid argument: $argument1"
fi