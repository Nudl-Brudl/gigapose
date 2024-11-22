import os

def check_permissions(filepath):
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"{filepath} does not exist.")
        return
    
    # Check read permissions
    if os.access(filepath, os.R_OK):
        print(f"Read access to {filepath} is allowed.")
    else:
        print(f"No read access to {filepath}.")

    # Check write permissions
    if os.access(filepath, os.W_OK):
        print(f"Write access to {filepath} is allowed.")
    else:
        print(f"No write access to {filepath}.")

    # Check execute permissions
    if os.access(filepath, os.X_OK):
        print(f"Execute access to {filepath} is allowed.")
    else:
        print(f"No execute access to {filepath}.")

# Example usage
filepath = 'gigaPose_datasets/datasets/hope/models/obj_000011.ply'
check_permissions(filepath)
