#!/usr/bin/env fish

# Test server script for fish shell

# Exit if a command fails
function check_error
    if test $status -ne 0
        echo "Error: Command failed with status $status"
        exit 1
    end
end

# Get the directory where this script is located
set BASE_DIR (dirname (status --current-filename))
cd $BASE_DIR

# Check if virtual environment exists
if not test -d ./venv
    echo "Error: Virtual environment not found."
    echo "Please run setup_fish.fish first to create the environment and install packages."
    exit 1
end

# Activate the virtual environment
echo "Activating virtual environment..."
source ./venv/bin/activate.fish
check_error

# Make sure required packages are installed
echo "Verifying required packages..."
python -c "import flask; import torch; import PIL; print('All required packages are installed.')"
if test $status -ne 0
    echo "Error: Some required packages are missing."
    echo "Please run setup_fish.fish to install all dependencies."
    exit 1
end

# Start the server
echo "Starting the trash detection server..."
echo "Press Ctrl+C to stop the server."

set -x FLASK_APP server.py
python server.py &
set SERVER_PID $last_pid

# Give the server time to start
sleep 3

# Open the browser
echo "Opening browser to http://localhost:8080"
xdg-open http://localhost:8080 2>/dev/null
open http://localhost:8080 2>/dev/null

echo
echo "Testing Instructions:"
echo "1. Use the web interface to upload an image or use the webcam"
echo "2. Try both server-side and browser-side processing"
echo "3. Check the detection results for trash items"
echo "4. Test with different types of waste (plastic, paper, metal, etc.)"
echo
echo "Press Ctrl+C to stop the server when done testing"

# Wait for user to press Ctrl+C
trap "kill $SERVER_PID; exit 0" SIGINT
sleep 99999 # Just to keep the script running 