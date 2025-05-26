#!/bin/bash
set -e

# Setup environment variables
export PYTHONPATH=/mmdetection:$PYTHONPATH

# Check the first argument for specific commands
if [ "$1" = "train" ]; then
  # Remove the first argument (train)
  shift
  # Run training script with remaining arguments
  echo "Running training with arguments: $@"
  python /mmdetection/tools/train.py "$@"

elif [ "$1" = "test" ]; then
  # Remove the first argument (test)
  shift
  # Run testing script with remaining arguments
  echo "Running testing with arguments: $@"
  python /mmdetection/tools/test.py "$@"

elif [ "$1" = "help" ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
  echo "MMDetection Docker Container"
  echo "Usage options:"
  echo "  - train: Run training with specified config"
  echo "  - test: Run testing with specified config and checkpoint"

elif [ "$#" -eq 0 ]; then
  # No arguments provided, run the default CMD from Dockerfile
  exec python /mmdetection/tools/train.py --help

else
  # Any other command is executed directly
  echo "Running custom command: $@"
  exec "$@"
fi
