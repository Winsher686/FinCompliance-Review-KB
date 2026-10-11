"""部署与工程脚手架回归测试（Docker / Compose / CI / 依赖清单）。

守的是「能用一条命令把它跑起来」这件事本身：

- `Dockerfile`：多阶段构建、非 root 运行、健康检查、CUDA 版本与 torch wheel 对齐；
- `docker-compose.yml`：4 个依赖服务 + 3 个应用服务齐全，且都带健康检查与依赖顺序；
- `docker-compose.gpu.yml`：三个应用服务都能叠加上 GPU 预留；
- `.github/workflows/ci.yml`：lint / test / docker 三个 job 存在，test 必须真的跑 pytest 与 ruff；
- `requirements-ci.txt`：**不许**混入 torch / mineru 等重型包——否则 CI 会从 2 分钟变成十几分钟。

这些文件的价值全在「还在不在、还能不能解析」，所以用测试把它们钉住，而不是靠人记得别删。
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pytest

yaml = pytest.importorskip("yaml", reason="需要 pyyaml 解析部署配置（见 requirements-ci.txt）")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

APP_SERVICES = ("app-import", "app-query", "app-review")
DEP_SERVICES = ("etcd", "minio", "milvus", "neo4j", "mongo")


def _read(relative: str) -> str:
    with open(os.path.join(_ROOT, relative), "r", encoding="utf-8") as f:
        return f.read()


def _load_yaml(relative: str):
    return yaml.safe_load(_read(relative))


# ==================== Dockerfile ====================

def test_dockerfile_is_multi_stage_and_gpu_aware():
    text = _read("Dockerfile")
    # 多阶段：依赖层与运行层分开，改业务代码不用重装 torch
    assert text.count("\nFROM ") >= 2, "应为多阶段构建（builder + runtime）"
    assert "AS builder" in text and "AS runtime" in text
    # GPU 基镜像，且 CUDA 版本与 requirements 里的 torch+cu118 对齐
    assert "nvidia/cuda" in text
    assert "CUDA_VERSION=11.8.0" in text, "CUDA 版本必须与 torch+cu118 wheel 对齐"
    assert "cu118" in _read("knowledge/requirements.txt")


def test_dockerfile_runs_as_non_root_with_healthcheck():
    text = _read("Dockerfile")
    assert "\nUSER appuser" in text, "容器不应以 root 运行"
    assert "useradd" in text
    assert "HEALTHCHECK" in text and "/health" in text
    # 三个服务端口都要暴露
    for port in ("8000", "8001", "8002"):
        assert port in text


def test_dockerfile_default_command_is_review_service():
    text = _read("Dockerfile")
    assert "review_router:create_app" in text
    assert "--factory" in text, "路由模块用的是 create_app 工厂，必须带 --factory"


def test_dockerignore_excludes_secrets_and_caches():
    text = _read(".dockerignore")
    for pattern in (".env", "knowledge/.env", "__pycache__/", ".git/"):
        assert pattern in text, f".dockerignore 应排除 {pattern}"
    # 示例文档要保留（导入演示依赖它）
    assert "sample_docs" not in text


# ==================== docker-compose ====================

def test_compose_declares_all_services():
    compose = _load_yaml("docker-compose.yml")
    services = compose["services"]
    for name in DEP_SERVICES + APP_SERVICES:
        assert name in services, f"compose 缺少服务：{name}"
    assert set(services) == set(DEP_SERVICES + APP_SERVICES)


def test_compose_every_service_has_healthcheck():
    compose = _load_yaml("docker-compose.yml")
    for name, svc in compose["services"].items():
        assert "healthcheck" in svc, f"{name} 缺健康检查：编排无法判断「起来没有」"
        assert svc["healthcheck"].get("test"), f"{name} 健康检查命令为空"


def test_compose_app_services_depend_on_healthy_deps():
    compose = _load_yaml("docker-compose.yml")
    for name in APP_SERVICES:
        depends = compose["services"][name].get("depends_on") or {}
        assert depends, f"{name} 未声明依赖顺序"
        for dep, cfg in depends.items():
            assert cfg.get("condition") == "service_healthy", (
                f"{name} 应在 {dep} 健康后再启动（否则首次建连必失败）"
            )


def test_compose_app_services_point_to_internal_hostnames():
    """容器内必须用服务名互相访问，不能用 127.0.0.1（那是容器自己）。"""
    compose = _load_yaml("docker-compose.yml")
    env = compose["services"]["app-review"]["environment"]
    assert env["MILVUS_URL"] == "http://milvus:19530"
    assert env["NEO4J_URI"] == "bolt://neo4j:7687"
    assert env["MONGO_URL"] == "mongodb://mongo:27017"
    assert env["MINIO_ENDPOINT"] == "minio:9000"


def test_compose_has_persistent_volumes_and_private_network():
    compose = _load_yaml("docker-compose.yml")
    volumes = compose.get("volumes") or {}
    for name in ("milvus_data", "neo4j_data", "mongo_data", "minio_data", "etcd_data"):
        assert name in volumes, f"缺少持久化卷：{name}"
    assert compose.get("networks"), "应显式声明网络，避免用默认网桥"


def test_gpu_override_adds_reservation_to_all_app_services():
    base = _load_yaml("docker-compose.yml")
    gpu = _load_yaml("docker-compose.gpu.yml")
    for name in APP_SERVICES:
        devices = gpu["services"][name]["deploy"]["resources"]["reservations"]["devices"]
        assert devices and devices[0]["capabilities"] == ["gpu"], f"{name} 未叠加 GPU 预留"
        # 基础清单里不能有 GPU 预留，否则没有 GPU 的机器直接起不来
        assert "deploy" not in base["services"][name]


# ==================== CI ====================

def test_ci_workflow_jobs_and_triggers():
    ci = _load_yaml(".github/workflows/ci.yml")
    # YAML 会把裸 on 解析成布尔 True
    triggers = ci.get("on", ci.get(True))
    assert triggers, "CI 未声明触发条件"
    assert "push" in triggers and "pull_request" in triggers

    jobs = ci["jobs"]
    for job in ("lint", "test", "docker"):
        assert job in jobs, f"CI 缺少 job：{job}"
    assert jobs["test"]["strategy"]["matrix"]["python-version"], "测试应做多版本矩阵"


def test_ci_test_job_actually_runs_tests_and_lint():
    ci = _load_yaml(".github/workflows/ci.yml")
    runs = "\n".join(
        step.get("run", "") for step in ci["jobs"]["test"]["steps"]
    )
    assert "pytest knowledge/test/compliance" in runs
    assert "run_eval.py" in runs, "评测指标应进 CI（分数回退要能被拦下）"

    lint_runs = "\n".join(step.get("run", "") for step in ci["jobs"]["lint"]["steps"])
    assert "ruff check ." in lint_runs


def test_ci_uses_lightweight_requirements():
    """CI 必须用最小依赖集，否则每次跑都要装几个 GB 的 torch。"""
    ci = _read(".github/workflows/ci.yml")
    assert "requirements-ci.txt" in ci
    assert "requirements-dev.txt" not in ci, "dev 依赖会连带安装 torch/mineru，拖垮 CI"


def test_ci_gpu_job_is_opt_in():
    """GPU 冒烟必须默认跳过（托管 runner 没有 GPU），通过仓库变量开启。"""
    ci = _load_yaml(".github/workflows/ci.yml")
    job = ci["jobs"]["gpu-smoke"]
    assert "self-hosted" in job["runs-on"]
    assert "ENABLE_GPU_RUNNER" in str(job["if"])
    assert job["timeout-minutes"] > 0


def test_requirements_ci_stays_lightweight():
    """只检查「生效的依赖行」，注释里为了解释原因提到 torch 是可以的。"""
    lines = [
        line.strip() for line in _read("knowledge/requirements-ci.txt").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    declared = " ".join(lines).lower()

    for heavy in ("torch", "mineru", "sentence-transformers", "milvus-lite"):
        assert heavy not in declared, f"requirements-ci.txt 不应包含重型依赖 {heavy}"
    for needed in ("pytest", "fastapi", "uvicorn", "langgraph", "pyyaml"):
        assert needed in declared, f"requirements-ci.txt 缺少离线测试必需依赖 {needed}"
    # 不应通过 -r 间接把重型清单引回来
    assert "-r requirements.txt" not in declared


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
