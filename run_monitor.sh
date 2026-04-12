#!/bin/bash
cd "/Users/jasonjohnson/Documents/Claude/Projects/Basilect Engine/basilect-engine-main"
python3 monitor.py 2>&1
echo ""
echo "Monitor exited with code $?"
echo "Press enter to close..."
read
