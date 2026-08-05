import os
from pathlib import Path
import subprocess
import sys

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
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "${SBATCH_CAPTURE}"\n'
        'printf "%s\\n" "${SYNONIM_BENCHMARK_GUROBI_TIME_LIMIT:-}" > "${SBATCH_CAPTURE}.gurobi"\n',
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SBATCH_CAPTURE": str(capture_path),
        "SYNONIM_BENCHMARK_MILP_TIME": slurm_limit,
        "SYNONIM_BENCHMARK_OUTPUT_DIR": str(tmp_path / "outputs"),
        "SYNONIM_BENCHMARK_RUN_ID": "pytest",
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
    assert "--export=ALL" in arguments
    assert (tmp_path / "sbatch-args.txt.gurobi").read_text(encoding="utf-8").strip() == (
        expected_gurobi_limit
    )


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
        "SYNONIM_BENCHMARK_OUTPUT_DIR": str(tmp_path / "outputs"),
        "SYNONIM_BENCHMARK_RUN_ID": "pytest",
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
        "SYNONIM_BENCHMARK_RUN_ID": "pytest",
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
    run_id_index = arguments.index("--run-id")
    assert arguments[run_id_index + 1] == "pytest"
    prefix_index = arguments.index("--summary-prefix")
    assert arguments[prefix_index + 1] == "binary_milp_k10"


def test_batch_job_preserves_grouped_sizes_and_custom_profile_path(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture_path = tmp_path / "python-args.txt"
    write_executable(
        bin_dir / "python",
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "${PYTHON_CAPTURE}"\n',
    )
    profiles_path = tmp_path / "profiles.json"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PYTHON_CAPTURE": str(capture_path),
        "SYNONIM_BENCHMARK_PROFILE": "large",
        "SYNONIM_BENCHMARK_STRATEGY": "heuristic",
        "SYNONIM_BENCHMARK_CONSORTIA_SIZES": "1,2,3,5",
        "SYNONIM_BENCHMARK_PROFILES_PATH": str(profiles_path),
        "SYNONIM_BENCHMARK_RUN_ID": "pytest",
    }

    subprocess.run([str(JOB_SCRIPT)], cwd=tmp_path, env=env, check=True)

    arguments = capture_path.read_text(encoding="utf-8").splitlines()
    size_values = [
        arguments[index + 1]
        for index, value in enumerate(arguments)
        if value == "--consortia-size"
    ]
    assert size_values == ["1", "2", "3", "5"]
    profiles_index = arguments.index("--profiles-path")
    assert arguments[profiles_index + 1] == str(profiles_path)


def test_full_large_submission_is_isolated_complete_and_collision_free(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture_path = tmp_path / "sbatch-calls.txt"
    write_executable(
        bin_dir / "python",
        f"#!/usr/bin/env bash\nexec {sys.executable!s} \"$@\"\n",
    )
    write_executable(
        bin_dir / "sbatch",
        "#!/usr/bin/env bash\n"
        "{\n"
        "  printf 'CALL\\n'\n"
        "  printf 'run=%s\\n' \"${SYNONIM_BENCHMARK_RUN_ID}\"\n"
        "  printf 'sizes=%s\\n' \"${SYNONIM_BENCHMARK_CONSORTIA_SIZES}\"\n"
        "  printf 'strategy=%s\\n' \"${SYNONIM_BENCHMARK_STRATEGY}\"\n"
        "  printf 'output=%s\\n' \"${SYNONIM_BENCHMARK_OUTPUT_DIR}\"\n"
        "  printf 'gurobi=%s\\n' \"${SYNONIM_BENCHMARK_GUROBI_TIME_LIMIT:-}\"\n"
        "  printf '%s\\n' \"$@\"\n"
        "} >> \"${SBATCH_CAPTURE}\"\n",
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SBATCH_CAPTURE": str(capture_path),
        "SYNONIM_BENCHMARK_OUTPUT_DIR": str(tmp_path / "outputs"),
        "SYNONIM_BENCHMARK_RUN_ID": "large-pytest",
    }

    subprocess.run(
        [str(SUBMIT_SCRIPT), "large"],
        cwd=tmp_path,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    lines = capture_path.read_text(encoding="utf-8").splitlines()
    assert lines.count("CALL") == 85
    assert set(line for line in lines if line.startswith("run=")) == {"run=large-pytest"}
    expected_sizes = "1,2,3,4,5,7,10,12,15,17,20,25,30"
    assert f"sizes={expected_sizes}" in lines
    job_names = [line for line in lines if line.startswith("--job-name=")]
    assert len(job_names) == len(set(job_names)) == 85
    assert set(line for line in lines if line.startswith("gurobi=") and line != "gurobi=") == {
        "gurobi=69120"
    }
    result_dirs = set(line for line in lines if line.startswith("output="))
    assert result_dirs == {f"output={tmp_path}/outputs/large-pytest/results"}

    manifest_path = tmp_path / "outputs" / "large-pytest" / "manifest.json"
    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "large-pytest"
    assert len(manifest["jobs"]) == 481
    assert len(manifest["jobs"]) * manifest["expected_scenarios"] == 3848
    assert len({job["job_key"] for job in manifest["jobs"]}) == 481
