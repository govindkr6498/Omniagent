import logging
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import SQLDatabaseToolkit
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine, text
import difflib
import os

class SQLAgentTool:
    def __init__(self, db_path: str, openai_api_key: str):
        self.logger = logging.getLogger("SQLAgentTool")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.db_path = db_path
        self.openai_api_key = openai_api_key
        self._setup()

    def _setup(self):
        db_uri = f"sqlite:///{self.db_path}"
        self.db = SQLDatabase.from_uri(db_uri)
        self.engine = create_engine(db_uri)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=self.openai_api_key)
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.tools = self.toolkit.get_tools()
        self.tool_names = [t.name for t in self.tools]
        with self.engine.connect() as conn:
            rows = conn.execute(text("SELECT DISTINCT area_c FROM properties")).all()
            self.areas = [row[0] for row in rows]
        self.logger.info(f"Loaded {len(self.areas)} distinct area_c values from DB")
        template = """
You are a helpful, expert data analyst agent for a real estate sales chatbot. You must answer the user's question using the available tools, and always reply in a friendly, conversational style suitable for a chat with a potential customer.

The main table in the database is called 'properties'.
Each row represents a property unit available for sale or rent. The columns are:
- id: unique property record id
- ownerid: owner id
- name: property code
- area_c: area name (e.g., 'Al Khawaneej', 'Bur Dubai', etc)
- area_code_c: area code (numeric)
- plot_number_c: plot number
- project_name_c: project name
- project_type_c: project type (e.g., Residential, Personal, Commercial)
- building_number_c: building number
- tower_name_c: tower name
- unit_no_c: unit number
- unit_type_c: unit type (e.g., Studio Apartment, 3 Bed Room Apartment, 2 Bed Room Apartment)
- unit_price_c: price of the unit
- district_c: district name (e.g., Dubai)

Column values may have typos or case differences. Always try to match user input to the closest valid value in the data, even if there are minor spelling or case mistakes. If a value is not found, explain this politely.

**IMPORTANT: You must always follow the exact ReAct format below. Do not skip any step. If you skip any required step, your answer will be rejected.**

When you answer, always explain the result in a friendly, conversational way, not just as raw data. If the user asks for something that does not exist, let them know clearly.

{tools}

Use this format:

Question: the input question you must answer  
Thought: always think about what to do  
Action: the action to take, must be one of [{tool_names}]  
Action Input: the input to the action  
Observation: the result of the action  
... (repeat Thought/Action/Action Input/Observation as needed)  
Thought: I now know the final answer  
Final Answer: include the complete detailed answer in a friendly, conversational style — for example, list each property with its code, type, price, and project name. Do NOT skip or summarize.


**IMPORTANT INSTRUCTIONS:**

- Do not skip or merge any steps.
- Even when you are ready to answer, always write `Thought: I now know the final answer` before writing the final answer.
- Then, and only then, write `Final Answer:` with a friendly, conversational response.
- Do NOT write `Final Answer:` directly after an `Observation:`. Always insert a `Thought:` line before the final answer.


Do not wrap SQL queries in ``` or markdown formatting. Keep output clean. Be concise.

Begin!

Question: {input}
{agent_scratchpad}
"""
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
        )
        self.agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent, tools=self.tools, verbose=True,
            handle_parsing_errors=True, max_iterations=10, early_stopping_method="generate"
        )

    def fuzzy_match(self, user_val: str, choices: list[str], cutoff: float = 0.6) -> str:
        matches = difflib.get_close_matches(user_val, choices, n=1, cutoff=cutoff)
        return matches[0] if matches else user_val

    def fix_area(self, q: str) -> str:
        import re
        m = re.search(r"in ([A-Za-z ]+)", q, re.IGNORECASE)
        if m:
            user_area = m.group(1).strip()
            canonical = self.fuzzy_match(user_area, self.areas)
            self.logger.info(f"Mapped area '{user_area}' to canonical '{canonical}'")
            return re.sub(re.escape(m.group(1)), canonical, q, flags=re.IGNORECASE)
        return q

    def query(self, question: str) -> str:
        self.logger.info(f"Received SQL question: {question}")
        clean_question = self.fix_area(question)
        self.logger.info(f"Cleaned question: {clean_question}")
        try:
            result = self.agent_executor.invoke({"input": clean_question})
            self.logger.info("SQL query processed successfully")
            return result["output"]
        except Exception as e:
            self.logger.error("SQL agent error", exc_info=True)
            return f"SQL agent error: {str(e)}"
