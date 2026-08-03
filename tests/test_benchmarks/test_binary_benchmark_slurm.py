import os
from pathlib import Path
import subprocess

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMIT_SCRIPT = REPO_ROOT / "benchmarks" / "slurm" / "submit_binary_profile.sh"
JOB_SCRIPT = REPO_ROOT / "benchmarks" / "slurm" / "run_binary_job.sbatch"


def write_executable(path, contents):
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


@pytest.mark.parametrize(
    ("slurm_limit", "expected_gurobi_limit"),
    [
        ("120", "5760"),
        ("10:30", "504"),
        ("10:00:00", "28800"),
        ("1-02", "74880"),
        ("1-02:30", "76320"),
        ("1-02:30:15", "76332"),
        ("1-00:00:00", "69120"),
        ("0:01", "0.8"),
    ],
)
def test_milp_submission_uses_eighty_percent_of_slurm_time(
    tmp_path, slurm_limit, expected_gurobi_limit
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture_path = tmp_path / "sbatch-args.txt"
    write_executable(
        bin_dir / "python",
        "#!/usr/bin/env bash\nprintf 'milp k=10 acp=1 amr=0\\n1 job(s)\\n'\n",
    )
    write_executable(
        bin_dir / "sbatch",
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "${SBATCH_CAPTURE}"\n',
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SBATCH_CAPTURE": str(capture_path),
        "SYNONIM_BENCHMARK_MILP_TIME": slurm_limit,
    }

    subprocess.run(
        [str(SUBMIT_SCRIPT), "large", "--strategy", "milp"],
        cwd=tmp_path,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    arguments = capture_path.read_text(encoding="utf-8").splitlines()
    assert f"--time={slurm_limit}" in arguments
    exports = next(
        argument for argument in arguments if argument.startswith("--export=")
    )
    assert f"SYNONIM_BENCHMARK_GUROBI_TIME_LIMIT={expected_gurobi_limit}" in exports


def test_invalid_milp_slurm_time_fails_before_submission(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_executable(
        bin_dir / "python",
        "#!/usr/bin/env bash\nprintf 'milp k=10 acp=1 amr=0\\n1 job(s)\\n'\n",
    )
    write_executable(bin_dir / "sbatch", "#!/usr/bin/env bash\nexit 99\n")
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SYNONIM_BENCHMARK_MILP_TIME": "UNLIMITED",
    }

    result = subprocess.run(
        [str(SUBMIT_SCRIPT), "large", "--strategy", "milp"],
        cwd=tmp_path,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "Invalid finite SLURM time limit 'UNLIMITED'" in result.stderr


def test_batch_job_passes_gurobi_time_limit_to_runner(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture_path = tmp_path / "python-args.txt"
    write_executable(
        bin_dir / "python",
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "${PYTHON_CAPTURE}"\n',
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PYTHON_CAPTURE": str(capture_path),
        "SYNONIM_BENCHMARK_PROFILE": "large",
        "SYNONIM_BENCHMARK_STRATEGY": "milp",
        "SYNONIM_BENCHMARK_CONSORTIA_SIZES": "10",
        "SYNONIM_BENCHMARK_GUROBI_TIME_LIMIT": "69120",
    }

    subprocess.run(
        [str(JOB_SCRIPT)],
        cwd=tmp_path,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    arguments = capture_path.read_text(encoding="utf-8").splitlines()
    time_limit_index = arguments.index("--time-limit")
    assert arguments[time_limit_index + 1] == "69120"
