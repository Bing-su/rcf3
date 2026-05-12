import nox

python = ["3.11", "3.14", "3.14t"]
venv_backend = "uv"


@nox.session(
    python=python,
    venv_backend=venv_backend,
)
def test(session: nox.Session) -> None:
    session.install(".[test]")
    session.run("pytest", "-v", *session.posargs)
