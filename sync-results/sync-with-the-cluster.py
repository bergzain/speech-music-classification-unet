import paramiko
import os
import subprocess
from dotenv import load_dotenv
from tqdm import tqdm
import re

# Load environment variables from .env file
load_dotenv()

# SSH configuration from environment variables
ssh_host = os.getenv('SSH_HOST')
ssh_port = int(os.getenv('SSH_PORT', 22))
ssh_user = os.getenv('SSH_USER')
ssh_password = os.getenv('SSH_PASSWORD')

# Directories
remote_results_dir = '/netscratch/zhazzouri/experiments/'
local_results_dir = '/Users/zainhazzouri/projects/Master-thesis-experiments/'

def sync_folders():
    """
    Synchronize folders between a remote SSH cluster and the local machine.

    This function connects to a remote SSH cluster using the provided SSH
    credentials and uses `rsync` to synchronize the contents of the remote
    results directory with the local results directory. After synchronization,
    it deletes the files from the remote server that were successfully copied
    to the local machine.

    Environment Variables:
    - SSH_HOST: The hostname or IP address of the SSH server.
    - SSH_PORT: The port number of the SSH server (default is 22).
    - SSH_USER: The username for SSH authentication.
    - SSH_PASSWORD: The password for SSH authentication.

    Directories:
    - remote_results_dir: The path to the results directory on the remote SSH server.
    - local_results_dir: The path to the results directory on the local machine.
    """
    # Create SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(ssh_host, port=ssh_port, username=ssh_user, password=ssh_password)
    except paramiko.AuthenticationException:
        print("Authentication failed, please verify your credentials.")
        return
    except paramiko.SSHException as sshException:
        print(f"Unable to establish SSH connection: {sshException}")
        return
    except Exception as e:
        print(f"Exception in connecting to SSH server: {e}")
        return

    try:
        # Use rsync to synchronize folders, remove source files, and capture the output
        rsync_command = f'rsync -avz --checksum --progress -e "sshpass -p {ssh_password} ssh -o StrictHostKeyChecking=no" {ssh_user}@{ssh_host}:{remote_results_dir}/ {local_results_dir}/'
        process = subprocess.Popen(rsync_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Initialize progress bar
        total_files = 0
        progress_bar = None
        changes_detected = False

        for line in process.stdout:
            # Print the line to show transfer speed and other details
            print(line, end='')
            # Check if there are changes to sync
            if not changes_detected and re.search(r'Number of files transferred: \d+', line):
                changes_detected = True
            # Parse the total number of files
            if 'to-check=' in line:
                match = re.search(r'to-check=(\d+)/(\d+)', line)
                if match:
                    files_left = int(match.group(1))
                    total_files = int(match.group(2))
                    if progress_bar is None:
                        progress_bar = tqdm(total=total_files)
                    progress_bar.n = total_files - files_left
                    progress_bar.refresh()

        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, rsync_command)
        if not changes_detected:
            print("No changes to sync.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running rsync: {e}")
    finally:
        # Close SSH connection
        ssh.close()
        if progress_bar:
            progress_bar.close()

if __name__ == '__main__':
    sync_folders()
