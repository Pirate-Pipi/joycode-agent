import docker
import docker.api
import logging
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple

JOYCODE_ROOT = Path(__file__).parent.parent
MAX_RUNTIME_PARALLELISM = 8


def get_issue_image_name(problem_id: str, workspace: Path) -> str:
    """Fetch a docker image for the issue."""
    issue_key = problem_id.replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{issue_key}:latest"


def set_volume_permissions(container_id, volume_path: Path):
    # Make sure we can read the volume
    # Docker is running as root, we may be running as JoyCode.
    my_uid = os.getuid()
    my_gid = os.getgid()
    logging.info(f"Fixing permissions for {volume_path} to {my_uid}:{my_gid}")
    env = os.environ.copy()

    # Only chmod directories that we can actually modify (avoid system directories like /tmp, /)
    modifiable_parents = []
    for parent in volume_path.parents:
        parent_str = parent.as_posix()
        # Skip system directories that we can't modify
        if parent_str in ['/', '/tmp', '/var', '/usr', '/etc', '/sys', '/proc']:
            continue
        # Only include directories under /tmp/workspace or similar user-writable paths
        if '/tmp/workspace' in parent_str or '/home' in parent_str:
            modifiable_parents.append(parent_str)

    if modifiable_parents:
        try:
            subprocess.check_call(
                [
                    "chmod",
                    "a+rx",
                ]
                + modifiable_parents,
                env=env,
            )
        except subprocess.CalledProcessError as e:
            logging.warning(f"Failed to chmod {modifiable_parents}: {e}")
            # Don't raise here, as this is not critical for functionality
    else:
        logging.info("No modifiable parent directories found, skipping chmod")
    # Change the owner to the current user
    try:
        container_out = subprocess.check_output(
            [
                "chown",
                "-R",
                f"{my_uid}:{my_gid}",
                volume_path.as_posix(),
            ],
            env=env,
            text=True,
            errors="backslashreplace",
        )
        logging.debug(container_out)
    except subprocess.CalledProcessError as e:
        logging.warning(f"Failed to chown {volume_path}: {e}")
        raise


def start_container(workspace: Path, problem_id: str, semaphore: Any) -> str:
    """Start a docker container for the issue."""
    terminate_runtime_pod(f"swe-bench-agent.{problem_id}")
    image_name = get_issue_image_name(problem_id, workspace)

    client = docker.from_env()
    logging.info(f"Pulling image {image_name}")   

    try:
        client.images.get(image_name)
    except docker.errors.ImageNotFound:
        with semaphore:
            logging.info(f"Pulling image {image_name} from remote")
            remote_image_name = 'docker.1ms.run/' + image_name

            # Add retry mechanism
            max_retries = 3

            for attempt in range(max_retries):
                try:
                    logging.info(f"Attempt {attempt + 1}/{max_retries}: Pulling {remote_image_name}")

                    # Use the high-level API to pull the image to avoid low-level API compatibility issues
                    try:
                        client.images.pull(remote_image_name)
                        logging.info(f"Successfully pulled {remote_image_name}")
                    except docker.errors.ImageNotFound:
                        # If the high-level API fails, try using subprocess to call the docker command
                        logging.info(f"High-level API failed, trying subprocess approach")
                        pull_cmd = ["docker", "pull", remote_image_name]
                        result = subprocess.run(pull_cmd, capture_output=True, text=True, timeout=300)
                        if result.returncode != 0:
                            raise docker.errors.APIError(f"Docker pull failed: {result.stderr}")
                        logging.info(f"Successfully pulled {remote_image_name} using subprocess")

                    # Tag the image
                    logging.info(f"Tagging image {remote_image_name} as {image_name}")
                    remote_image = client.images.get(remote_image_name)
                    remote_image.tag(image_name)

                    # Clean up the remote tag
                    try:
                        client.images.remove(remote_image_name)
                        logging.info(f"Removed remote tag {remote_image_name}")
                    except docker.errors.APIError as e:
                        logging.warning(f"Failed to remove remote tag {remote_image_name}: {e}")

                    logging.info(f"Finished pulling image {image_name}")
                    break  # Success, exit retry loop

                except (docker.errors.APIError, subprocess.TimeoutExpired, Exception) as e:
                    logging.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        # Final attempt failed, raise exception
                        logging.error(f"Failed to pull image {remote_image_name} after {max_retries} attempts")
                        raise docker.errors.ImageNotFound(f"Could not pull image {remote_image_name}: {e}")
                    else:
                        # Wait before retrying
                        wait_time = (attempt + 1) * 10  # Incremental backoff
                        logging.info(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)

    logging.info(f"Running docker run for {image_name} in {workspace}")

    # Safely handle symlink deletion
    symlink_path = workspace / problem_id
    if symlink_path.exists() or symlink_path.is_symlink():
        try:
            symlink_path.unlink(missing_ok=True)
        except (PermissionError, OSError) as e:
            logging.warning(f"Failed to remove symlink {symlink_path}: {e}")
            try:
                subprocess.run(['sudo', 'rm', '-f', str(symlink_path)], check=True)
                logging.info(f"Successfully removed symlink with sudo: {symlink_path}")
            except subprocess.CalledProcessError as sudo_e:
                logging.error(f"Failed to remove symlink even with sudo: {sudo_e}")
                unique_id = uuid.uuid4().hex[:8]
                new_path = workspace / f"{problem_id}_{unique_id}"
                logging.info(f"Using alternative path: {new_path}")
                symlink_path = new_path

    # Generate a unique container name
    container_unique_id = uuid.uuid4().hex[:8]
    container_name = f"swe-bench-agent.{problem_id}_{container_unique_id}"

    # Improved Docker startup command to ensure the git repository is properly initialized
    init_command = """bash -c '
        echo "=== Starting Git repository initialization ==="

        # Set Git configuration (ignore errors)
        echo "Setting up git configuration..."
        git config --global user.email "agent@example.com" || true
        git config --global user.name "SWE-Bench Agent" || true
        git config --global --add safe.directory /testbed || true
        git config --global init.defaultBranch main || true

        # Enter working directory
        cd /testbed || exit 1
        echo "Working directory: $(pwd)"
        echo "Directory listing: $(ls -la | head -10)"

        # Check Git repository status
        if [ -d ".git" ]; then
            echo "Existing .git directory found"

            # Check repository status
            echo "Checking repository status..."
            git status --porcelain || echo "Git status failed, but continuing..."

            # If there are uncommitted changes, commit them first
            if git status --porcelain 2>/dev/null | grep -q .; then
                echo "Found uncommitted changes, committing them..."
                git add . || echo "Git add failed, but continuing..."
                git commit -m "Auto-commit existing changes before agent start" || echo "Git commit failed, but continuing..."
            else
                echo "No uncommitted changes found"
            fi
        else
            echo "No .git directory found, initializing new repository..."
            git init || exit 1
            echo "Git repository initialized"

            # Add all files and create an initial commit
            echo "Adding all files to git..."
            git add . || exit 1
            echo "Creating initial commit..."
            git commit -m "Initial commit - SWE-bench repository setup" || exit 1
            echo "Initial commit created"
        fi

        # Create a marker commit to ensure there is a HEAD to compare
        echo "Creating agent start marker commit..."
        git commit --allow-empty -m "JoyCode agent start marker - $(date)" || echo "Marker commit failed, but continuing..."

        # Verify Git status
        echo "=== Final Git Status ==="
        git log --oneline -3 || echo "Git log failed"
        git status || echo "Git status failed"
        git rev-parse HEAD || echo "HEAD parsing failed"

        echo "=== Git setup complete, keeping container alive ==="

        # Keep the container running
        trap "echo Container shutting down; exit 0" SIGTERM SIGINT
        while true; do
            sleep 60
            # Periodically check whether the container should continue running
            if [ ! -f /testbed/.git/HEAD ]; then
                echo "Git repository seems corrupted, exiting..."
                exit 1
            fi
        done
    '"""

    with semaphore:
        logging.info(f"Starting run for {image_name}")

        # The SWE-bench image already contains the full repository; we do not need to bind over /testbed
        # Instead, map the container's /testbed to a host path for later access
        container = client.containers.run(
            name=container_name,
            image=image_name,
            detach=True,
            # Do not override /testbed; let the container use the original code from the image
            # volumes={str(symlink_path): {'bind': '/testbed', 'mode': 'rw'}},
            command=init_command,
        )
        logging.info(f"Finished startup for {image_name}")

        # Create a symlink to copy contents from the container to the host (if needed)
        # This will run after container initialization
    
    container_id = container.id
    assert container_id is not None
    logging.info(f"Started container {container_id} for {problem_id}")

    # Wait for the container to start and initialize the Git repository
    logging.info("Waiting for container initialization...")
    max_wait_time = 30  # Reduce wait time to 30 seconds
    wait_interval = 3   # Check every 3 seconds

    container_ready = False
    for attempt in range(max_wait_time // wait_interval):
        try:
            # Check whether the container is still running
            container.reload()
            if container.status != 'running':
                logging.warning(f"Container {container_id} status: {container.status} (attempt {attempt + 1})")
                if container.status == 'exited':
                    # Fetch container logs for diagnosis
                    try:
                        logs = container.logs(tail=20).decode('utf-8', errors='ignore')
                        logging.error(f"Container exited. Last 20 lines of logs:\n{logs}")
                    except Exception as log_e:
                        logging.error(f"Failed to get container logs: {log_e}")
                    break
                time.sleep(wait_interval)
                continue

            # Check Git repository status (lenient check)
            result = container.exec_run("bash -c 'cd /testbed && pwd && ls -la .git && git rev-parse HEAD 2>/dev/null || echo NO_HEAD'")
            if result.exit_code == 0:
                output = result.output.decode().strip()
                logging.info(f"Container check passed (attempt {attempt + 1})")
                logging.debug(f"Container output: {output}")

                # Check whether a valid Git HEAD exists
                if 'NO_HEAD' not in output and len(output.split('\n')[-1]) >= 7:  # Git commit hash should be at least 7 chars
                    logging.info(f"Git repository ready with HEAD commit")
                    container_ready = True
                    break
                else:
                    logging.info(f"Git repository exists but no HEAD commit yet (attempt {attempt + 1})")
            else:
                error_output = result.output.decode().strip()
                logging.warning(f"Container check failed (attempt {attempt + 1}): {error_output}")

        except Exception as e:
            logging.warning(f"Failed to check container status (attempt {attempt + 1}): {e}")

        if attempt < (max_wait_time // wait_interval) - 1:
            time.sleep(wait_interval)

    if not container_ready:
        logging.warning(f"Container initialization may not be complete after {max_wait_time} seconds, but continuing...")
        # Do not raise; continue execution

    # Ensure host directory exists
    symlink_path.mkdir(parents=True, exist_ok=True)

    # Copy repository from container to host (for later host-side access)
    try:
        logging.info(f"Copying repository from container to {symlink_path}")
        copy_cmd = f"docker cp {container_id}:/testbed/. {symlink_path}/"
        result = subprocess.run(copy_cmd, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            logging.info("Successfully copied repository from container to host")
        else:
            logging.warning(f"Failed to copy repository from container: {result.stderr}")
    except Exception as e:
        logging.warning(f"Error copying repository from container: {e}")

    # Set permissions
    retry = True
    while True:
        try:
            set_volume_permissions(container_id, symlink_path)
            break
        except Exception as e:
            logging.warning(f"Failed to set permissions: {e}")
            if not retry:
                break
            retry = False
            time.sleep(5)

    return container_id


def remove_container_image(image_name: str) -> None:
    """Remove a docker image."""
    try:
        client = docker.from_env()
        client.images.remove(image=image_name, force=True)
        logging.info(f"Removed image {image_name}")
    except docker.errors.APIError as e:  # type: ignore
        logging.warning(f"Failed to remove image {image_name}: {e}")


def terminate_runtime_pod(container_id: str, remove_image: str = "") -> None:
    """Stop a docker container for the issue."""
    container = None
    try:
        client = docker.from_env()
        container = client.containers.get(container_id)
    except Exception as e:
        logging.info(f"Container {container_id} not found: {e}")

    if container:
        try:
            logging.info(f"Stopping container {container_id}")
            container.stop()
            logging.info(f"Stopped container {container_id}")
        except docker.errors.NotFound as e:  # type: ignore
            logging.warning(f"Failed to stop container {container_id}: {e}")
        except docker.errors.APIError as e:  # type: ignore
            logging.warning(f"Failed to stop container {container_id}: {e}")
        try:
            logging.info(f"Removing container {container_id}")
            container.remove()
            time.sleep(10)
            logging.info(f"Removed container {container_id}")
        except docker.errors.NotFound as e:  # type: ignore
            logging.warning(f"Failed to stop container {container_id}: {e}")
        except docker.errors.APIError as e:  # type: ignore
            logging.warning(f"Failed to stop container {container_id}: {e}")

    if remove_image:
        # Add a small delay to ensure container removal is complete
        time.sleep(5)
        remove_container_image(remove_image)


def configure_project_space(
    workspace: Path, problem_id: str, lock: Any, semaphore: Any
) -> Tuple[Dict[str, str], str]:
    """Setup the workspace for the agent."""
    env: Dict[str, str] = os.environ.copy()

    # Create a conda environment; we don't use it, but it protects the
    # agent's environment from changes.
    logging.debug(f"Creating conda enviroment in {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    # Multiple simultaneous conda installs are no good.
    with lock:
        subprocess.check_output(
            [
                "python3",
                "-m",
                "venv",
                str(workspace / "venv"),
            ]
        )

    env["ISSUE_ID"] = problem_id
    env["SWEBENCH_WORKSPACE"] = str(workspace)
    env["PATH"] = f"{workspace}/python_wrappers/bin:{workspace}/venv/bin" + (
        f":{env['PATH']}" if "PATH" in env else ""
    )
    env["PYTHONPATH"] = f"{JOYCODE_ROOT}" + (
        f":{env['PYTHONPATH']}" if "PYTHONPATH" in env else ""
    )
    for k, v in env.items():
        logging.debug(f"ENV {k}=={v}")

    # Copy the python wrapper into the workspace
    container_id = start_container(workspace, problem_id, semaphore)

    return env, container_id
