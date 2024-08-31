import pkg_resources
import subprocess

# Function to install a package
def install_package(package):
    subprocess.check_call(["pip", "install", package])

# Function to check the installed version of a package
def check_version(requirement):
    try:
        pkg = pkg_resources.require(requirement)
        print(f"{pkg[0].project_name}=={pkg[0].version}")
    except pkg_resources.DistributionNotFound:
        print(f"{requirement.split('==')[0]} is not installed")
    except pkg_resources.VersionConflict as e:
        print(f"Version conflict for {e.req.project_name}: {e.dist.version} is installed but {e.req} is required")

# Read the requirements.txt file
with open("requirements.txt", "r") as f:
    requirements = f.readlines()

# Process each requirement
for req in requirements:
    req = req.strip()
    if req:  # If the line is not empty
        install_package(req)
        check_version(req)
