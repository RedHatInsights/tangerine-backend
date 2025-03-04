#!/bin/bash

# Exit on any error
set -e

# Function to display messages
log() {
    echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*"
}

# Check if the current directory is a Git repository
if [ ! -d ".git" ]; then
    log "Error: This script must be run inside a Git repository."
    exit 1
fi

# Define the pre-commit hook content
PRE_COMMIT_CONTENT='
#!/bin/bash

# Exit on any error
set -e

log() {
    echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*"
}

log "Running black formatter..."
black --line-length=100 **/*.py || { 
    log "Code formatting issues detected. Please run black **/*.py --line-length=100 to fix them.";
    exit 1;
}

log "Running flake8 linter..."
flake8 --max-line-length=200 **/*.py || { 
    log "Linting issues detected. Please fix the issues reported by flake8.";
    exit 1;
}

log "Pre-commit checks passed"
'

# Write the pre-commit hook to .git/hooks/pre-commit
log "Setting up pre-commit hook..."
echo "$PRE_COMMIT_CONTENT" > .git/hooks/pre-commit

# Make the pre-commit hook executable
chmod +x .git/hooks/pre-commit

log "Pre-commit hook has been successfully set up!"