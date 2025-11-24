# Transferring Files to WatGPU via SCP

This guide explains how to transfer files from your local machine to the WatGPU server using SCP (Secure Copy Protocol).

## Prerequisites

Before transferring files, ensure you have:

- **SSH key generated and added to the remote machine**: Your public SSH key must be added to the WatGPU server's authorized keys
- **SSH config file setup**: Your SSH config should be configured for easy access to WatGPU

## Instructions

### Step 1: Verify Remote Directory Exists

Ensure the destination directory exists on the remote machine. If it doesn't exist, create it first:

```bash
ssh <remote_username>@watgpu.cs.uwaterloo.ca "mkdir -p <remote_path>"
```

### Step 2: Transfer Files

Use the following command to transfer files or directories:

```bash
scp -r <local_path> <remote_username>@watgpu.cs.uwaterloo.ca:<remote_path>
```

**Command breakdown:**
- `-r`: Recursive flag (required for directories)
- `<local_path>`: Path to the file or directory on your local machine
- `<remote_username>`: Your WatGPU username
- `<remote_path>`: Destination path on the remote machine


## Example usage: Uploading a large dataset to the remote machine

```bash
scp -r ~/Projects/diabetes/nocturnal-hypo-gly-prob-forecast/cache/data/awesome_cgm/aleppo/raw t3chan@watgpu.cs.uwaterloo.ca:/u4/t3chan/nocturnal-hypo-gly-prob-forecast/cache/data/awesome_cgm/aleppo/raw
```
