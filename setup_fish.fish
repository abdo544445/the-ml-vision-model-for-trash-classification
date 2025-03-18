#!/usr/bin/env fish

# Fish shell script to set up the trash detection environment

# Print heading
function print_header
    echo ""
    echo "=================================================="
    echo $argv[1]
    echo "=================================================="
end

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

# Activate virtual environment properly in fish
if test -d ./venv
    print_header "Activating existing virtual environment"
    if test -e ./venv/bin/activate.fish
        source ./venv/bin/activate.fish
    else
        echo "Error: Fish activation script not found in virtual environment"
        echo "Creating activate.fish script..."
        
        # Create fish activation script
        echo '# This file must be used with "source ./venv/bin/activate.fish" from fish
# you cannot run it directly

function deactivate --description "Exit virtual environment and return to normal shell environment"
    # reset old environment variables
    if test -n "$_OLD_VIRTUAL_PATH"
        set -gx PATH $_OLD_VIRTUAL_PATH
        set -e _OLD_VIRTUAL_PATH
    end

    if test -n "$_OLD_VIRTUAL_PYTHONHOME"
        set -gx PYTHONHOME $_OLD_VIRTUAL_PYTHONHOME
        set -e _OLD_VIRTUAL_PYTHONHOME
    end

    if test -n "$_OLD_FISH_PROMPT_OVERRIDE"
        functions -e fish_prompt
        set -e _OLD_FISH_PROMPT_OVERRIDE
        functions -c _old_fish_prompt fish_prompt
        functions -e _old_fish_prompt
    end

    set -e VIRTUAL_ENV
    if test "$argv[1]" != "nondestructive"
        # Self-destruct!
        functions -e deactivate
    end
end

# Unset irrelevant variables
deactivate nondestructive

set -gx VIRTUAL_ENV "$BASE_DIR/venv"

set -gx _OLD_VIRTUAL_PATH $PATH
set -gx PATH "$VIRTUAL_ENV/bin" $PATH

# Unset PYTHONHOME if set
if set -q PYTHONHOME
    set -gx _OLD_VIRTUAL_PYTHONHOME $PYTHONHOME
    set -e PYTHONHOME
end

if test -z "$VIRTUAL_ENV_DISABLE_PROMPT"
    # fish uses a function instead of an env var to generate the prompt.
    # Save the current fish_prompt function as the function _old_fish_prompt.
    functions -c fish_prompt _old_fish_prompt

    function fish_prompt
        # Save the return status of the last command
        set -l old_status $status
        printf "%s(%s) " (set_color normal) (basename "$VIRTUAL_ENV")
        # Restore the return status of the previous command
        echo "exit $old_status" | .
        # Output the original prompt
        _old_fish_prompt
    end

    set -gx _OLD_FISH_PROMPT_OVERRIDE "$VIRTUAL_ENV"
end' > ./venv/bin/activate.fish
        chmod +x ./venv/bin/activate.fish
        source ./venv/bin/activate.fish
    end
else
    print_header "Creating new virtual environment"
    python -m venv venv
    check_error
    source ./venv/bin/activate.fish
end

# Install required packages
print_header "Installing required packages"

set PACKAGES torch torchvision flask flask-cors pillow ultralytics

for package in $PACKAGES
    echo "Installing $package..."
    python -m pip install $package
    check_error
end

print_header "Environment setup complete!"
echo "You can now run the server with: python server.py"
echo "Then open http://localhost:8080 in your browser"
echo ""
echo "To start the server now, type: python server.py"
echo "" 