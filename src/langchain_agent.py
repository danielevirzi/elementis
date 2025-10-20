"""
LangChain Agent with Vigilance Tools

Creates an agent that can use the Vigilance environmental monitoring tools
to answer questions about fires and floods in Italy.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEndpoint

from langchain_tools import get_all_vigilance_tools

logger = logging.getLogger(__name__)


class VigilanceLangChainAgent:
    """
    LangChain agent with Vigilance environmental monitoring tools
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048
    ):
        """
        Initialize LangChain agent with tools
        
        Args:
            model_name: HuggingFace model to use (defaults to env/config)
            temperature: Model temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
        """
        logger.info("Initializing VigilanceLangChainAgent...")
        
        # Model settings
        self.model_name = model_name or os.getenv(
            "HUGGINGFACE_MODEL",
            "meta-llama/Llama-3.2-3B-Instruct"
        )
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize tools
        self.tools = get_all_vigilance_tools()
        logger.info(f"Loaded {len(self.tools)} Vigilance tools")
        
        # Initialize LLM
        self._init_llm()
        
        # Initialize agent
        self._init_agent()
        
        logger.info("VigilanceLangChainAgent initialized successfully")
    
    def _init_llm(self):
        """Initialize HuggingFace LLM via LangChain"""
        try:
            # Create HuggingFace endpoint (use directly for ReAct)
            self.llm = HuggingFaceEndpoint(
                repo_id=self.model_name,
                task="text-generation",
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                huggingfacehub_api_token=self.hf_token,
            )
            
            logger.info(f"Initialized LLM: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise
    
    def _init_agent(self):
        """Initialize LangChain ReAct agent with tools"""
        try:
            # Create ReAct prompt template
            template = """Sei un assistente esperto in monitoraggio ambientale italiano.

Hai accesso a strumenti per interrogare dati su:
- 🔥 **Incendi**: Dati satellitari FIRMS NASA (MODIS/VIIRS) con informazioni su incendi attivi in Italia
- 🌊 **Alluvioni**: Bollettini di criticità idrogeologica della Protezione Civile italiana

**Linee guida per l'uso degli strumenti:**

1. **Incendi (Hotspots)**:
   - Usa tools geografici (regione, provincia, comune) per query localizzate
   - Usa filtri intensità/confidenza per incendi significativi
   - Usa ricerca spaziale (distanza, bounding box) per aree specifiche
   - Usa statistiche per panoramiche generali

2. **Alluvioni (Floods)**:
   - Specifica sempre 'oggi' o 'domani' per il bulletin_type
   - Usa filtri rischio (Verde/Giallo/Arancione/Rosso) per allerte
   - Classi numeriche 0-3 per filtraggio preciso
   - Confronta bollettini per vedere cambiamenti nel rischio

3. **Livelli di rischio alluvione**:
   - Verde (0): Nessuna allerta
   - Giallo (1): Allerta ordinaria
   - Arancione (2): Allerta moderata
   - Rosso (3): Allerta elevata

**Formato delle risposte:**
- Sii conciso ma completo
- Cita numeri specifici (es: "15 incendi in Sicilia")
- Specifica la fonte dei dati (hotspots NASA, bollettino PC)
- Usa emoji per chiarezza (🔥 incendi, 🌊 alluvioni, ⚠️ allerte)
- Se non ci sono dati, dillo chiaramente

TOOLS:
------
You have access to the following tools:

{tools}

To use a tool, use the following format:

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

Thought: Do I need to use a tool? No
Final Answer: [your response here]

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

            prompt = PromptTemplate.from_template(template)
            
            # Create ReAct agent
            self.agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )
            
            # Create agent executor
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=10,
                max_execution_time=60,
            )
            
            logger.info("Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing agent: {e}")
            raise
    
    def query(self, question: str, chat_history: Optional[List] = None) -> Dict[str, Any]:
        """
        Query the agent with a question
        
        Args:
            question: User question in Italian
            chat_history: Optional chat history for context
        
        Returns:
            Dictionary with answer and metadata
        """
        try:
            logger.info(f"Processing query: {question}")
            
            # Prepare input
            agent_input = {
                "input": question,
                "chat_history": chat_history or []
            }
            
            # Run agent
            result = self.agent_executor.invoke(agent_input)
            
            return {
                "success": True,
                "question": question,
                "answer": result["output"],
                "intermediate_steps": result.get("intermediate_steps", [])
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "success": False,
                "question": question,
                "error": str(e)
            }
    
    def list_available_tools(self) -> List[Dict[str, str]]:
        """
        List all available tools
        
        Returns:
            List of dictionaries with tool info
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "args": str(tool.args_schema.schema()) if hasattr(tool, 'args_schema') else "None"
            }
            for tool in self.tools
        ]


# ============================================================================
# STANDALONE FUNCTIONS FOR EASY USAGE
# ============================================================================

def create_vigilance_agent(
    model_name: Optional[str] = None,
    temperature: float = 0.1
) -> VigilanceLangChainAgent:
    """
    Create a Vigilance LangChain agent
    
    Args:
        model_name: Optional HuggingFace model name
        temperature: Model temperature
    
    Returns:
        Initialized agent
    """
    return VigilanceLangChainAgent(
        model_name=model_name,
        temperature=temperature
    )


def query_vigilance_agent(question: str, agent: Optional[VigilanceLangChainAgent] = None) -> str:
    """
    Query the Vigilance agent (creates agent if not provided)
    
    Args:
        question: Question in Italian
        agent: Optional pre-initialized agent
    
    Returns:
        Answer string
    """
    if agent is None:
        agent = create_vigilance_agent()
    
    result = agent.query(question)
    
    if result["success"]:
        return result["answer"]
    else:
        return f"Errore: {result['error']}"


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create agent
    agent = create_vigilance_agent()
    
    # Example queries
    queries = [
        "Quanti incendi ci sono in Sicilia?",
        "Quali regioni hanno allerte arancioni o rosse oggi?",
        "Mostrami gli incendi entro 50 km da Roma",
        "Confronta le allerte di oggi con quelle di domani",
        "Quali zone hanno rischio idrogeologico elevato?",
    ]
    
    print("=" * 80)
    print("Vigilance LangChain Agent - Demo")
    print("=" * 80)
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 80}")
        print(f"Query {i}: {query}")
        print("=" * 80)
        
        result = agent.query(query)
        
        if result["success"]:
            print(f"\n✅ Answer:\n{result['answer']}")
        else:
            print(f"\n❌ Error: {result['error']}")
    
    print(f"\n{'=' * 80}")
    print("Demo completed!")
    print("=" * 80)
