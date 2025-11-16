import os
import json
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY missing in .env")

QA_JSON = "questions_and_answers.json"
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
LLM_MODEL_NAME = "gpt-4o-mini"

def convert_kb_to_doc(kb_path: str):
    """Convert JSON format into LangChain Documents"""
    with open(kb_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    docs = []
    for item in items:
        question = item.get("question", "")
        answer = item.get("answer", "")
        metadata = {"question": question}
        doc = Document(page_content=answer, metadata=metadata)
        docs.append(doc)

    return docs

def build_vector_store(docs, embeddings, persist_dir):
    """Initialize FAISS index"""
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print("Loading existing FAISS index from", persist_dir)
        vs = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Creating new FAISS index and saving to", persist_dir)
        vs = FAISS.from_documents(docs, embeddings)
        vs.save_local(persist_dir)
    return vs

def get_system_prompt():
    """Return the system prompt for the Coolblue support chatbot"""
    return """Je bent een vriendelijke en behulpzame Coolblue klantenservice medewerker. Je taak is om veelgestelde vragen van klanten te beantwoorden op basis van de beschikbare kennisbank.

BELANGRIJKE RICHTLIJNEN:
- Beantwoord vragen ALLEEN op basis van de verstrekte context uit de kennisbank
- Geef geen productadvies - focus uitsluitend op klantenservice en ondersteuningsvragen
- Als de context niet voldoende informatie bevat om de vraag te beantwoorden, wees dan eerlijk en verwijs de klant door
- Wees altijd beleefd, professioneel en klantvriendelijk
- Gebruik duidelijke, eenvoudige Nederlandse taal
- Houd antwoorden beknopt maar compleet

DOORVERWIJZING:
Als je de vraag niet kunt beantwoorden met de beschikbare informatie, zeg dan:
"Ik kan deze vraag helaas niet beantwoorden op basis van mijn beschikbare informatie. Voor persoonlijke hulp kun je contact opnemen met onze klantenservice op 010 798 8999. Zij helpen je graag verder!"

Wees altijd transparant over de beperkingen van je kennis."""

def create_rag_chain(retriever, llm):
    """Create a RAG chain"""
    
    system_prompt = get_system_prompt()
    
    template = """Context uit de kennisbank:
{context}

Klantvraag: {question}

Antwoord (gebaseerd op bovenstaande context):"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", template)
    ])
    
    rag_chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join([f"- {d.page_content}" for d in docs])),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def rag_answer(query, rag_chain, retriever):
    """Use the RAG chain to answer a query"""
    try:
        answer = rag_chain.invoke(query)
        docs_with_scores = retriever.vectorstore.similarity_search_with_score(query, k=4)
        
        return answer, docs_with_scores
    except Exception as e:
        print(f"Fout tijdens verwerking: {e}")
        return "Sorry, er is een fout opgetreden. Probeer het opnieuw of neem contact op met onze klantenservice op 010 798 8999.", []

def print_welcome_message():
    """Print welcome message with disclaimers"""
    print("=" * 70)
    print("     Welkom bij de Coolblue AI Chatbot!")
    print("=" * 70)
    print()
    print("LET OP: Dit is een AI-assistent die je helpt met veelgestelde vragen.")
    print("   - Deze assistent kan fouten maken")
    print("   - Je gegevens worden NIET opgeslagen")
    print("   - U kunt ook contact opnemen met onze klantenservice via 010 798 8999")
    print()
    print("Deze assistent helpt met klantenservice vragen, geen productadvies.")
    print()
    print("Type je vraag of 'stop' om te stoppen.")
    print("=" * 70)
    print()

def main():
    docs = convert_kb_to_doc(QA_JSON)
    print(f"Loaded {len(docs)} documents from {QA_JSON}")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, openai_api_key=OPENAI_API_KEY)

    vs = build_vector_store(docs, embeddings, FAISS_INDEX_PATH)

    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = ChatOpenAI(model_name=LLM_MODEL_NAME, openai_api_key=OPENAI_API_KEY, temperature=0.0)

    rag_chain = create_rag_chain(retriever, llm)

    print_welcome_message()

    while True:
        user_question = input("Jij: ").strip()
        if user_question.lower() in ("stop"):
            print("\nDe Coolblue AI Chatbot sessie is beëindigd. Tot ziens!")
            break

        if not user_question:
            continue

        answer, sources = rag_answer(user_question, rag_chain, retriever)
        
        print(f"\nCoolblue: {answer}\n")

        if sources:
            print("=" * 70)
            print("Debugging (Top-k gerelateerde vragen uit de kennisbank):")
            for i, (doc, score) in enumerate(sources, start=1):
                print(f"   {i}. {doc.metadata.get('question','N/A')} (Similarity Score: {score:.4f})")
        print("-" * 70)

if __name__ == "__main__":
    main()