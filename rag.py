from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

VECTORSTORE_PATH = "vectorstore"


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Завантаження векторного сховища (п. 5.2.7)
def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


# Створення пошукового компонента з налаштуванням параметрів (п. 5.2.7, 5.2.12)
def build_retriever(vectorstore, search_type="mmr", k=4, fetch_k=10, lambda_mult=0.5):
    if search_type == "mmr":
        search_kwargs = {
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult
        }
    else:
        search_kwargs = {"k": k}

    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )


# Системний промпт із правилами формування відповіді (п. 5.2.8)
prompt = ChatPromptTemplate.from_template(
    """
Ти — корпоративний AI-асистент ІТ-компанії TechNova Solutions, який відповідає на питання
співробітників на основі наданих внутрішніх регламентів та стандартів.

ПРАВИЛА:
1. Відповідай тільки на основі наведеного контексту.
2. Використовуй всю релевантну інформацію з контексту.
3. Формуй повну та змістовну відповідь, а не коротку відповідь одним реченням.
4. Формуй відповідь переважно зв’язним текстом. Використовуй списки лише для переліків, послідовностей дій або кількох чітко відокремлених пунктів.
5. Якщо в контексті справді немає достатньої інформації, скажи:
   "У наданих документах немає інформації з цього питання."
6. Не вигадуй і не додавай інформацію, якої немає в контексті.
7. Відповідай українською мовою.
8. Тон відповіді — професійний, чіткий і зрозумілий розробнику.

КОНТЕКСТ:
{context}

ПИТАННЯ:
{question}

ВІДПОВІДЬ:
"""
)


# Підключення мовної моделі (п. 5.2.9)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.2,
)


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def ask_rag(question, retriever):
    docs = retriever.invoke(question)

    context = format_docs(docs)

    messages = prompt.invoke({
        "context": context,
        "question": question
    })

    response = llm.invoke(messages)

    sources = sorted({
        doc.metadata.get("filename", "unknown")
        for doc in docs
    })

    return extract_text(response.content), sources, docs


# Тестування якості пошуку та генерації на різних типах запитів (п. 5.2.7, 5.2.11)
if __name__ == "__main__":
    vs = load_vectorstore()
    ret = build_retriever(vs, search_type="mmr", k=4)

    test_queries = [
        # 1. Питання за одним документом
        "Які вимоги до проведення Code Review та скільки схвалень потрібно?",
        # 2. Питання, що охоплює декілька документів (Git + Архітектура/Деплой)
        "Як відбувається процес від створення гілки для нової задачі до її автоматичного деплою на Staging?",
        # 3. Питання, інформації про яке немає в базі знань
        "Який розмір компенсації за абонемент у спортзал та курси англійської мови?"
    ]

    for i, q in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"ТЕСТОВИЙ ЗАПИТ №{i}: {q}")
        answer, src, found_docs = ask_rag(q, ret)
        print(f"Знайдено фрагментів ретривером: {len(found_docs)}")
        print(f"\nВІДПОВІДЬ:\n{answer}")
        print(f"\nДЖЕРЕЛА: {', '.join(src)}")