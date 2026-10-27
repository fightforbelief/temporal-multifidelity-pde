import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from temporal_mf.env_audit import assert_windows_repo, repo_root_from_here, write_environment_report

def main() -> None:
    repo = repo_root_from_here()
    assert_windows_repo(repo)
    info = write_environment_report(repo, repo / "reports" / "01_REMOTE_ENVIRONMENT.md")
    print(json.dumps({k: info[k] for k in ("hostname", "executable", "cuda_available", "gpu_name", "repo")}))

if __name__ == "__main__":
    main()
