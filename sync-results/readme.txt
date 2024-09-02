
# README

## Project Overview

This script is designed to synchronize files between a remote server and a local directory using SSH and `rsync`. The script securely connects to the remote server using the SSH protocol, retrieves the contents of a specified remote directory, and then syncs those contents to a local directory.

## Prerequisites

1. **Python 3.x**: Ensure you have Python 3.x installed on your system.
2. **Paramiko**: A Python library used to handle SSH connections.
3. **rsync**: A utility for efficiently transferring and synchronizing files across systems.
4. **dotenv**: A Python library used to load environment variables from a `.env` file.

### Installation of Required Python Packages

You can install the required Python packages using `pip`:

```bash
pip install paramiko python-dotenv
```

### Ensuring `rsync` is Installed

`rsync` should be installed on both the local and remote systems. On most Linux distributions, you can install it using:

```bash
sudo apt-get install rsync
```

## Environment Variables

The script reads configuration values from a `.env` file located in the same directory as the script. The `.env` file should contain the following environment variables:

```plaintext
SSH_HOST=your_remote_host_address
SSH_PORT=22  # Optional, defaults to 22
SSH_USER=your_ssh_username
SSH_KEY_PATH=/path/to/your/private/key
```

## Directories

- **Remote Directory**: The directory on the remote server containing the files to be synchronized. Set this in the script using `remote_results_dir`.
- **Local Directory**: The directory on the local machine where files will be copied. Set this in the script using `local_results_dir`.

## Running the Script

1. Create a `.env` file in the same directory as the script with the required environment variables.
2. Modify the `remote_results_dir` and `local_results_dir` variables in the script to match your directories.
3. Run the script:

```bash
python sync_script.py
```

## Error Handling

- **AuthenticationException**: Raised if SSH authentication fails. Ensure your SSH credentials are correct.
- **SSHException**: Raised if the SSH connection fails.
- **subprocess.CalledProcessError**: Raised if the `rsync` command fails. This usually indicates an issue with the command syntax or the remote directory path.

## Notes

- Make sure your SSH key is configured with the correct permissions.
- The script uses `paramiko` to establish an SSH connection and `rsync` for the actual file synchronization.
