# ---- config for folder copying ----
DEST_FOLDERS="rupertwilliams@10.8.202.72:~/QGIS/'Fire Interval Data'"  # where the folders will land
SOURCE_BASE="/fire_analysis_data/data/"  # base directory containing folders

# List of specific folders to copy (add/remove as needed)
FOLDERS_TO_COPY=(
    "burn_probabilities/compressed" # 1G
    "st/compressed" # 0.25G
    "hex_grid/cleaned" # Nothing
    "nlcd/cleaned" # ~40 GB
    "eco_regions/rasters" # Nothing
)

# Function to copy folders
copy_folders() {
    # Sanity check: does SOURCE_BASE exist and is it a dir?
    [ -d "$SOURCE_BASE" ] || { echo "SOURCE_BASE '$SOURCE_BASE' is not a directory"; exit 1; }
    
    # Extract hostname from DEST_FOLDERS for SSH connection
    HOST=$(echo "$DEST_FOLDERS" | cut -d':' -f1)
    
    # Setup SSH connection multiplexing (reuse single connection)
    SSH_CONTROL_PATH="/tmp/ssh-control-%r@%h:%p"
    
    echo "Setting up SSH connection to $HOST..."
    echo "You will be prompted for your password once, you have 10 seconds to enter it."
    
    # Establish master SSH connection with host networking
    echo "Attempting to establish SSH master connection..."
    ssh -o ControlMaster=yes -o ControlPath="$SSH_CONTROL_PATH" -o ControlPersist=10m -o StrictHostKeyChecking=no -N "$HOST" &
    SSH_PID=$!
    
    # Wait a moment for connection to establish
    echo "Waiting for SSH connection to establish..."
    sleep 10
    
    # Test if the connection is working
    echo "Testing SSH connection..."
    if ssh -o ControlPath="$SSH_CONTROL_PATH" -O check "$HOST" 2>/dev/null; then
        echo "✓ SSH master connection established successfully"
    else
        echo "✗ SSH master connection failed"
        echo "Trying without connection multiplexing..."
        # Kill the background process
        kill $SSH_PID 2>/dev/null
        # Try direct rsync without multiplexing
        USE_MULTIPLEXING=false
    fi
    
    echo "Starting folder copy operation..."
    echo "Source: $SOURCE_BASE"
    echo "Destination: $DEST_FOLDERS"
    echo "Folders to copy: ${FOLDERS_TO_COPY[*]}"
    echo "----------------------------------------"
    
    for folder in "${FOLDERS_TO_COPY[@]}"; do
        source_path="$SOURCE_BASE$folder"
        
        # Check if folder exists
        if [ -d "$source_path" ]; then
            echo "Copying folder: $source_path"
            # Use rsync for robust folder copying with automatic directory creation
            if [ "$USE_MULTIPLEXING" != "false" ]; then
                # Use SSH control socket to reuse connection
                rsync -avz --progress --mkpath -e "ssh -o ControlPath=$SSH_CONTROL_PATH" "$source_path/" "$DEST_FOLDERS/$folder/"
            else
                # Use direct SSH without multiplexing
                rsync -avz --progress --mkpath "$source_path/" "$DEST_FOLDERS/$folder/"
            fi
            
            if [ $? -eq 0 ]; then
                echo "✓ Successfully copied: $source_path"
            else
                echo "✗ Failed to copy: $source_path"
            fi
        else
            echo "⚠ Warning: Folder does not exist: $source_path"
        fi
        echo "----------------------------------------"
    done
    
    # Clean up SSH connection
    echo "Cleaning up SSH connection..."
    if [ "$USE_MULTIPLEXING" != "false" ]; then
        ssh -o ControlPath="$SSH_CONTROL_PATH" -O exit "$HOST" 2>/dev/null
    fi
    
    echo "Folder copy operation completed."
}

copy_folders 