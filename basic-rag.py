import os
import openai
from llama_index.core import VectorStoreIndex
from llama_index.core import ServiceContext
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from trulens_eval import Tru
from utils import get_prebuilt_trulens_recorder



openaikey = os.environ["OPENAI_API_KEY"]

documents = SimpleDirectoryReader(
    input_files=["file.pdf"]
).load_data()

document = Document(text="\n\n".join([doc.text for doc in documents]))

llm = OpenAI(temperature=0.1)
service_context=ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_documents([document], service_context=service_context)

query_engine = index.as_query_engine()

response = query_engine.query("what is agility")

print(str(response))

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

tru = Tru()
tru.reset_database()

tru_recorder = get_prebuilt_trulens_recorder(query_engine,
                                             app_id="Direct Query Engine")

with tru_recorder as recording:
    for question in eval_question:
        response = query_engine.query(question)

records, feedback = tru.get_records_and_feedback(app_ids=[])

# launches on http://localhost:8501/
tru.run_dashboard()