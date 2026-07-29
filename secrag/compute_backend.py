"""Pluggable "run a command on a compute node and block until it finishes" backend.

Split out from secrag/remote_embed.py's file-staging logic (scp-ing inputs up and
outputs back), which stays the same regardless of how a job actually executes. Today
that's SLURM sbatch on a shared cluster (SlurmBackend) — a different environment might
run the command directly over a blocking SSH exec, submit to a different scheduler, or
call a cloud batch API. Swapping that in only means writing a new ComputeBackend, not
touching remote_embed.py's request/response plumbing.
"""
import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path


class ComputeBackendError(Exception):
    pass


class ComputeBackend(ABC):
    @abstractmethod
    def run(self, command: str, job_name: str) -> None:
        """Run `command` on the compute node and block until it finishes.

        Raises ComputeBackendError if the command exits non-zero, or if the backend
        itself fails (can't submit, times out waiting, etc).
        """


class SlurmBackend(ComputeBackend):
    """Submits `command` as an sbatch job over SSH and polls `squeue` for completion.

    Uses `squeue` rather than `sacct`/`sacctmgr` to check status and exit code — on this
    cluster the accounting daemon (slurmdbd) isn't reachable from compute nodes, but
    squeue (live job-queue state) works fine. Exit code is instead captured by appending
    an `echo $? > ...` sentinel to the submitted script and reading it back once the job
    has left the queue.
    """

    def __init__(
        self,
        host: str,
        scratch_dir: str,
        gres: str = "gpu:1",
        time_limit: str = "00:30:00",
        poll_interval: float = 10.0,
        timeout: float = 1800.0,
    ):
        self.host = host
        self.scratch_dir = scratch_dir
        self.gres = gres
        self.time_limit = time_limit
        self.poll_interval = poll_interval
        self.timeout = timeout

    def run(self, command: str, job_name: str) -> None:
        job_suffix = uuid.uuid4().hex[:8]
        remote_script = f"{self.scratch_dir}/{job_name}_{job_suffix}.sbatch"
        remote_log = f"{self.scratch_dir}/{job_name}_{job_suffix}.log"
        remote_exit_file = f"{self.scratch_dir}/{job_name}_{job_suffix}.exitcode"

        script = (
            "#!/bin/bash\n"
            f"#SBATCH --job-name={job_name}\n"
            f"#SBATCH --output={remote_log}\n"
            f"#SBATCH --time={self.time_limit}\n"
            f"#SBATCH --gres={self.gres}\n"
            # Without this, sbatch can schedule the job onto a different node in the pool
            # (confirmed: jobs landed on watgpu1109 instead of watgpu108) where the venv at
            # `scratch_dir` silently fails to import anything — its python3 binary is a bare
            # symlink to that node's own /usr/bin/python3, which has no knowledge of the
            # venv's site-packages, so the job dies with a ModuleNotFoundError instead of an
            # obviously environment-related error. Pinning to the submission host guarantees
            # the venv actually exists where the job runs.
            f"#SBATCH --nodelist={self.host}\n"
            f"{command}\n"
            f"echo $? > {remote_exit_file}\n"
        )

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            local_script = Path(tmpdir) / "job.sbatch"
            local_script.write_text(script)
            self._scp_to(local_script, remote_script)

        slurm_job_id = self._ssh(f"sbatch --parsable {remote_script}").strip()

        deadline = time.monotonic() + self.timeout
        while self._ssh(f"squeue -j {slurm_job_id} -h", check=False).strip():
            if time.monotonic() > deadline:
                self._ssh(f"scancel {slurm_job_id}", check=False)
                raise ComputeBackendError(
                    f"SLURM job {slurm_job_id} ({job_name}) timed out after {self.timeout}s"
                )
            time.sleep(self.poll_interval)

        exit_code = self._ssh(f"cat {remote_exit_file} 2>/dev/null", check=False).strip()
        self._ssh(f"rm -f {remote_script} {remote_log} {remote_exit_file}", check=False)
        if exit_code != "0":
            raise ComputeBackendError(
                f"SLURM job {slurm_job_id} ({job_name}) failed (exit code: {exit_code or 'unknown'})"
            )

    def _ssh(self, remote_command: str, check: bool = True) -> str:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=15", self.host, remote_command],
            capture_output=True, text=True,
        )
        if check and result.returncode != 0:
            raise ComputeBackendError(f"ssh command failed: {result.stderr}")
        return result.stdout

    def _scp_to(self, local_path: Path, remote_path: str) -> None:
        result = subprocess.run(
            ["scp", "-o", "ConnectTimeout=15", str(local_path), f"{self.host}:{remote_path}"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise ComputeBackendError(f"scp failed: {result.stderr}")
