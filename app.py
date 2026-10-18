import os
import streamlit as st
from dotenv import load_dotenv

# Завантаження змінних середовища (.env)
load_dotenv()

try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass
    
from rag import load_vectorstore, build_retriever, ask_rag

# Налаштування сторінки
st.set_page_config(
    page_title="TechNova RAG Assistant",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Корпоративний RAG-асистент TechNova")
st.write("Задавайте питання на основі внутрішніх регламентів та стандартів компанії.")

# Бічна панель для дослідження параметрів пошуку (п. 5.2.7, 5.2.12)
with st.sidebar:
    st.header("⚙️ Параметри пошуку (Retriever)")
    search_type = st.selectbox(
        "Спосіб пошуку (search_type)",
        options=["mmr", "similarity"],
        index=0,
        help="MMR забезпечує баланс між релевантністю та різноманітністю фрагментів. Similarity шукає найближчі за векторною відстанню."
    )
    k = st.slider(
        "Кількість фрагментів (k)",
        min_value=1,
        max_value=8,
        value=4,
        help="Скільки текстових фрагментів передавати в контекст мовної моделі."
    )
    if search_type == "mmr":
        fetch_k = st.slider("Кількість кандидатів (fetch_k)", min_value=5, max_value=20, value=10)
        lambda_mult = st.slider(
            "Коефіцієнт різноманітності (lambda_mult)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="0.0 — максимальна різноманітність, 1.0 — максимальна схожість."
        )
    else:
        fetch_k = 10
        lambda_mult = 0.5


# Кешування завантаження векторної бази
@st.cache_resource
def get_vectorstore():
    return load_vectorstore()


vectorstore = get_vectorstore()
retriever = build_retriever(
    vectorstore,
    search_type=search_type,
    k=k,
    fetch_k=fetch_k,
    lambda_mult=lambda_mult
)

# Історія чату в сесії
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown("**Джерела:**")
            for src in msg["sources"]:
                st.write(f"- `{src}`")

# Поле введення запиту користувача
question = st.chat_input("Введіть ваше запитання...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Пошук у документах та формування відповіді..."):
            answer, sources, docs = ask_rag(question, retriever)

        st.markdown(answer)

        no_info_phrase = "У наданих документах немає інформації з цього питання"
        display_sources = [] if no_info_phrase in answer else sources

        if display_sources:
            st.markdown("**Джерела:**")
            for source in display_sources:
                st.write(f"- `{source}`")

        with st.expander(f"🔍 Переглянути знайдені текстові фрагменти ({len(docs)} шт.)"):
            for idx, doc in enumerate(docs, 1):
                st.markdown(f"**Фрагмент #{idx}** | Файл: `{doc.metadata.get('filename')}`")
                st.caption(doc.page_content)
                st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": display_sources
    })