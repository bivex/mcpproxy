import subprocess
import sys
import os
from pathlib import Path

def run_command(command_args, use_shell=False):
    """Runs a shell command (or direct process) and prints its output."""
    try:
        if isinstance(command_args, list):
            printable_command = ' '.join(command_args)
        else:
            printable_command = command_args

        print(f"Running command: {printable_command}")
        result = subprocess.run(command_args, shell=use_shell, check=True, capture_output=True) # Removed text=True and encoding here for raw bytes
        print("STDOUT:")
        print(result.stdout.decode('cp866', errors='replace')) # Decode explicitly
        if result.stderr:
            print("STDERR:")
            print(result.stderr.decode('cp866', errors='replace')) # Decode explicitly
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {printable_command}")
        print("STDOUT:")
        print(e.stdout.decode('cp866', errors='replace'))
        print("STDERR:")
        print(e.stderr.decode('cp866', errors='replace'))
        return False
    except FileNotFoundError:
        print(f"Command not found: {command_args[0] if isinstance(command_args, list) else command_args.split(' ')[0]}. Please ensure it is installed and in your PATH.")
        return False

def main():
    print("Starting mcpproxy installation...")

    # 1. Check for uv installation
    print("Checking for 'uv' installation...")
    if not run_command(["uv", "--version"]) : # Use list for shell=False
        print("\n'uv' is not found. Please install 'uv' first. You can usually do this by running:")
        print("  curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("Or if you have pipx: pipx install uv")
        print("After installing, please restart your terminal and run this script again.")
        sys.exit(1)

    # 2. Create virtual environment
    venv_path = ".venv"
    if not os.path.exists(venv_path):
        print(f"Creating virtual environment in {venv_path}...")
        if not run_command(["uv", "venv"]): # Use list for shell=False
            print("\nFailed to create virtual environment. Please check for permission issues or if the directory is in use.")
            print("You might need to manually delete the '.venv' directory if it partially exists and try again.")
            sys.exit(1)
    else:
        print(f"Virtual environment already exists at {venv_path}.")

    # 3. Install dependencies
    print("Installing dependencies...")
    packages = "pydantic numpy bm25s mcp fastmcp"

    if sys.platform == "win32":
        # Use uv directly to install packages into the virtual environment
        command_args = ["uv", "pip", "install", "--python", str(Path(venv_path) / "Scripts" / "python.exe")]
        command_args.extend(packages.split())
        
        if not run_command(command_args): # Call with list of args, default use_shell=False
            print("\nFailed to install dependencies. Please check the error messages above.")
            sys.exit(1)

    else: # Linux/macOS
        # Use uv directly to install packages into the virtual environment
        command_args = ["uv", "pip", "install", "--python", str(Path(venv_path) / "bin" / "python")]
        command_args.extend(packages.split())
        
        if not run_command(command_args): # Call with list of args, default use_shell=False
            print("\nFailed to install dependencies. Please check the error messages above.")
            sys.exit(1)

    print("\nInstallation complete!")
    print("To activate the virtual environment and run the proxy, use:")
    if sys.platform == "win32":
        print(f"  .\\{venv_path}\\Scripts\\activate")
        print(f"  python C:\\Users\\Admin\\Desktop\\Dev\\mcpproxy\\main.py")
    else:
        print(f"  source {venv_path}/bin/activate")
        print(f"  python main.py") # Assuming main.py is in the root and activated env is used.

if __name__ == "__main__":
    main() 
