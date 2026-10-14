from typer.testing import CliRunner

from rag_lab.cli import app
from rag_lab.memory import Memory


def test_history_reads_messages_without_runtime(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'history.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    memory = Memory(url)
    session = memory.create()
    memory.save_turn(session, 0, "Question", "Answer", {})
    memory.close()

    def no_runtime(*args, **kwargs):
        raise AssertionError("Просмотр истории не должен запускать RAG")

    monkeypatch.setattr("rag_lab.cli.Runtime", no_runtime)
    result = CliRunner().invoke(app, ["history", "--session", session])
    assert result.exit_code == 0, result.output
    assert result.output.index("user: Question") < result.output.index("assistant: Answer")
    assert '[history.done]' in result.output
    assert '"messages": 2' in result.output
    assert session in result.output


def test_history_unknown_id_fails_without_success_log(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'history.db'}")
    result = CliRunner().invoke(app, ["history", "--session", "missing"])
    assert result.exit_code == 1
    assert "[history.error]" in result.output
    assert "[history.done]" not in result.output
