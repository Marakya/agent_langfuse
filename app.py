import streamlit as st

from rag_lab.config import RagConfig, Settings
from rag_lab.memory import Memory
from rag_lab.progress import progress
from rag_lab.runtime import Runtime

st.set_page_config(page_title="RAG Chat", layout="wide")
st.title("RAG Chat")
st.caption("Ответы по вашей базе знаний с учётом истории диалога.")


@st.cache_resource
def get_memory():
    progress("streamlit.start", status="Подключение к истории диалогов")
    memory = Memory(Settings().database_url)
    progress("streamlit.ready", conversations=len(memory.conversations()))
    return memory


@st.cache_resource
def get_runtime(config_json: str):
    progress("streamlit.rag.start")
    runtime = Runtime(Settings(), RagConfig.model_validate_json(config_json))
    progress("streamlit.rag.ready", config=runtime.index.config.name)
    return runtime


memory = get_memory()
with st.sidebar:
    st.header("Диалоги")
    if st.button("Новый диалог", use_container_width=True):
        st.session_state["conversation"] = memory.create()
    conversations = memory.conversations()
    if not conversations:
        st.session_state["conversation"] = memory.create()
        conversations = memory.conversations()
    ids = [item["id"] for item in conversations]
    titles = {item["id"]: item["title"] for item in conversations}
    if st.session_state.get("conversation") not in ids:
        st.session_state["conversation"] = ids[0]
    session_id = st.selectbox(
        "Продолжить диалог",
        ids,
        key="conversation",
        format_func=lambda value: f"{titles[value]} · {value[:8]}",
    )
    st.caption(f"ID: {session_id}")
    st.info("История сохраняется в базе данных. После перезапуска выберите прежний диалог.")

_, history = memory.history(session_id)
progress("streamlit.history.done", session_id=session_id, messages=len(history))


def show_sources(details):
    if not details.get("contexts"):
        return
    with st.expander("Источники ответа"):
        for number, context in enumerate(details["contexts"], 1):
            page = f" · стр. PDF {context['page']}" if context.get("page") else ""
            st.markdown(f"**[{number}] {context['source']}{page}**")
            st.text(context["text"])


for message in history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message["details"])

question = st.chat_input("Задайте вопрос по базе знаний")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    try:
        config = RagConfig.read(Settings().rag_config)
        runtime = get_runtime(config.model_dump_json())
        with st.spinner("Ищу в базе знаний…"):
            runtime.rag.chat(memory, session_id, question)
        runtime.telemetry.flush()
    except Exception as error:
        progress("streamlit.error", session_id=session_id, error=str(error))
        st.error(f"Не удалось получить ответ: {error}")
        st.caption(
            "Проверьте настройки .env и выполните uv run rag-chat ingest. Вопрос можно повторить."
        )
    else:
        st.rerun()
