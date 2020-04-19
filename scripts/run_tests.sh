#!/bin/bash

echo "Running Initialization Test..."
../bin/tests/test_initialization
echo "Running Element Access Test..."
../bin/tests/test_elementAccess
echo "Running Synchronization Test..."
../bin/tests/test_sync

