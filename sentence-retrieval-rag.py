import openai
import os
from llama_index.core import VectorStoreIndex
from llama_index.core import ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from trulens_eval import Tru
from utils import build_sentence_window_index
from utils import get_sentence_window_query_engine
from utils import get_prebuilt_trulens_recorder


openaikey = os.environ["OPENAI_API_KEY"]

documents = SimpleDirectoryReader(
    input_files=["file.pdf"]
).load_data()

document = Document(text="\n\n".join([doc.text for doc in documents]))


llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

sentence_index = build_sentence_window_index(
    document,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="sentence_index"
)

eval_question=["How does the Agile Practice Guide define the difference between definable work and high-uncertainty work?",
    "What are the four values of the Agile Manifesto, and how do they influence agile project management practices?",
    "Explain the relationship between Lean, the Kanban Method, and Agile approaches as described in the guide.",
    "Describe the characteristics of predictive, iterative, incremental, and agile life cycles as outlined in the guide.",
    "What are the key responsibilities of a servant leader in an agile environment, according to the guide?",
    "How does the guide suggest implementing agile practices within a traditional organizational structure?",
    "What factors should be considered when tailoring agile approaches to fit specific project needs?",
    "Discuss the concept of 'hybrid life cycles' and give an example of how a project might combine agile and predictive elements.",
    "How does the guide address the role and value of a project manager in an agile setting?",
    "Describe the process and importance of creating an agile environment for project teams as per the guide's recommendations."]

sentence_window_engine = get_sentence_window_query_engine(sentence_index)

window_response = sentence_window_engine.query(
    "What is Agility?"
)
print(str(window_response))

tru = Tru()
tru.reset_database()

tru_recorder_sentence_window = get_prebuilt_trulens_recorder(
    sentence_window_engine,
    app_id = "Sentence Window Query Engine"
)

for question in eval_question:
    with tru_recorder_sentence_window as recording:
        response = sentence_window_engine.query(question)
        print(question)
        print(str(response))

tru.get_leaderboard(app_ids=[])
tru.run_dashboard()
