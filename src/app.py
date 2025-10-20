"""
Main Gradio application for Elementis AI Agent
"""

import logging
import os
import warnings

import gradio as gr
from dotenv import load_dotenv

from agent import Agent
from utils import setup_logging

warnings.filterwarnings("ignore", message=".*flash_attn.*")
warnings.filterwarnings("ignore", message=".*Using PyTorch native attention.*")
logging.getLogger("transformers_modules").setLevel(logging.ERROR)

load_dotenv()
logger = setup_logging()

agent = None


def initialize_agent():
    """Initialize the agent with configuration"""
    global agent
    if agent is not None:
        logger.info("Agent already initialized")
        return "✅ Agente inizializzato correttamente!"
    
    try:
        logger.info("Initializing Elementis Agent...")
        agent = Agent()
        logger.info("Agent initialized successfully")
        return "✅ Agente inizializzato correttamente!"
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        return f"❌ Errore inizializzando l'agente: {str(e)}"


def process_query(query, history, workflow_status, enable_thinking):
    """Process user query and return response with workflow updates"""
    if agent is None:
        return "⚠️ Agente non initializzato. Per favore ricarica l'applicazione.", "❌ Non inizializzato", get_pipeline_html("idle")
    
    try:
        logger.info(f"Processing query: {query} (thinking mode: {enable_thinking})")
        
        result = agent.process(query, enable_thinking=enable_thinking)
        
        # Extract the answer from the result dictionary
        if isinstance(result, dict):
            response = result.get('answer', str(result))
            query_type = result.get('query_type', 'unknown')
            method = result.get('method', 'unknown')
            num_chunks = result.get('num_chunks', 0)
            tool_name = result.get('tool_name', None)
            tool_input = result.get('tool_input', None)
            
            # Create detailed workflow summary
            workflow_summary = "✅ **Query Completata**\n\n"
            
            # Stage 1: Routing
            workflow_summary += "**1. 🔄 Routing**\n"
            if query_type == 'RAG':
                workflow_summary += "   ✓ Tipo: RAG (Ricerca Documenti)\n\n"
            elif query_type == 'TOOL':
                workflow_summary += "   ✓ Tipo: TOOL (Accesso Dati)\n\n"
            elif query_type == 'NONE':
                workflow_summary += "   ✓ Tipo: FUORI AMBITO\n\n"
            elif query_type == 'DIRECT':   
                workflow_summary += "   ✓ Tipo: RICERCA NON NECESSARIA\n\n"
            else:
                workflow_summary += f"   ✓ Tipo: {query_type}\n\n"
            
            # Stage 2: Retrieval/Tool (if applicable)
            if tool_name and tool_input:
                # TOOL was called - show tool details
                workflow_summary += "**2. 🔧 Tool Execution**\n"
                workflow_summary += f"   ✓ Tool: `{tool_name}`\n"
                workflow_summary += f"   ✓ Input: `{tool_input}`\n\n"
            elif num_chunks > 0:
                # RAG was used - show retrieval details
                workflow_summary += "**2. 📚 Retrieval**\n"
                workflow_summary += f"   ✓ Metodo: {method}\n"
                workflow_summary += f"   ✓ Recuperati: {num_chunks} chunks\n\n"
            
            # Stage 3: Generation
            workflow_summary += "**3. 🤖 Generazione**\n"
            workflow_summary += "   ✓ Modello: Phi-4-mini-instruct\n"
            workflow_summary += f"   ✓ Lunghezza: {len(response)} caratteri\n"
            
            # Determine pipeline status based on query type
            if num_chunks > 0:
                pipeline_html = get_pipeline_html("complete_with_retrieval")
            else:
                pipeline_html = get_pipeline_html("complete_no_retrieval")
        else:
            response = str(result)
            workflow_summary = "✅ Complete"
            pipeline_html = get_pipeline_html("complete_no_retrieval")
            
        logger.info("Query processed successfully")
        return response, workflow_summary, pipeline_html
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        error_workflow = f"❌ **Error**\n\n{str(e)}"
        return f"❌ Error: {str(e)}", error_workflow, get_pipeline_html("error")


def get_pipeline_html(stage="idle"):
    """Generate HTML for pipeline status visualization"""
    
    # Define opacity and color based on stage
    if stage == "idle":
        router_opacity = "1"
        router_color = "#666"
        retrieval_opacity = "0.3"
        retrieval_color = "#ccc"
        generation_opacity = "0.3"
        generation_color = "#ccc"
    elif stage == "complete_with_retrieval":
        router_opacity = "1"
        router_color = "#4CAF50"
        retrieval_opacity = "1"
        retrieval_color = "#4CAF50"
        generation_opacity = "1"
        generation_color = "#4CAF50"
    elif stage == "complete_no_retrieval":
        router_opacity = "1"
        router_color = "#4CAF50"
        retrieval_opacity = "0.3"
        retrieval_color = "#ccc"
        generation_opacity = "1"
        generation_color = "#4CAF50"
    elif stage == "error":
        router_opacity = "1"
        router_color = "#f44336"
        retrieval_opacity = "0.3"
        retrieval_color = "#ccc"
        generation_opacity = "0.3"
        generation_color = "#ccc"
    else:
        router_opacity = "1"
        router_color = "#666"
        retrieval_opacity = "0.3"
        retrieval_color = "#ccc"
        generation_opacity = "0.3"
        generation_color = "#ccc"
    
    return f"""
    <div style="display: flex; justify-content: space-around; align-items: center; padding: 10px; background: #f5f5f5; border-radius: 8px; margin-bottom: 10px;">
        <div style="text-align: center;">
            <div style="font-size: 24px; opacity: {router_opacity};">🔄</div>
            <div style="font-size: 11px; color: {router_color};">Router</div>
        </div>
        <div style="font-size: 16px; color: #ccc;">→</div>
        <div style="text-align: center;">
            <div style="font-size: 24px; opacity: {retrieval_opacity};">📚</div>
            <div style="font-size: 11px; color: {retrieval_color};">Retrieval</div>
        </div>
        <div style="font-size: 16px; color: #ccc;">→</div>
        <div style="text-align: center;">
            <div style="font-size: 24px; opacity: {generation_opacity};">🤖</div>
            <div style="font-size: 11px; color: {generation_color};">Generazione</div>
        </div>
    </div>
    """


def create_interface():
    """Create Gradio interface"""
    
    # Custom CSS for smaller example buttons and bigger avatars
    custom_css = """
    #example-buttons button {
        font-size: 12px !important;
        padding: 6px 10px !important;
        min-height: 32px !important;
    }
    .avatar-container img {
        width: 48px !important;
        height: 48px !important;
    }
    .message-wrap .avatar-container {
        width: 48px !important;
        height: 48px !important;
    }
    """
    
    with gr.Blocks(
        title="Elementis - Agente AI per Catastrofi Naturali", 
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:
        gr.Markdown(
            """
            # 🌍 Elementis - Agente AI per Catastrofi Naturali
            
            Poni domande su catastrofi naturali, analizza dati ambientali e ottieni informazioni dai documenti.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Conversazione",
                    height=500,
                    show_label=True,
                    avatar_images=("https://api.dicebear.com/9.x/fun-emoji/svg?seed=Eden&backgroundColor=71cf62&mouth=cute", 
                                   "https://api.dicebear.com/9.x/bottts-neutral/svg?seed=Katherine&backgroundColor=1e88e5"),
                    type="messages"
                )
                
                with gr.Row():
                    query_input = gr.Textbox(
                        label="La Tua Domanda",
                        placeholder="Chiedi informazioni su alluvioni, incendi o dati sulle catastrofi naturali...",
                        lines=2,
                        scale=4
                    )
                    submit_btn = gr.Button("Invia", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("Pulisci", size="sm")

            
            with gr.Column(scale=1):
                gr.Markdown("###  Workflow dell'Agente")
                
                # Pipeline progress indicator
                pipeline_status = gr.HTML(
                    value="""
                    <div style="display: flex; justify-content: space-around; align-items: center; padding: 10px; background: #f5f5f5; border-radius: 8px; margin-bottom: 10px;">
                        <div style="text-align: center;">
                            <div style="font-size: 24px;">🔄</div>
                            <div style="font-size: 11px; color: #666;">Router</div>
                        </div>
                        <div style="font-size: 16px; color: #ccc;">→</div>
                        <div style="text-align: center;">
                            <div style="font-size: 24px; opacity: 0.3;">📚</div>
                            <div style="font-size: 11px; color: #ccc;">Retrieval</div>
                        </div>
                        <div style="font-size: 16px; color: #ccc;">→</div>
                        <div style="text-align: center;">
                            <div style="font-size: 24px; opacity: 0.3;">🤖</div>
                            <div style="font-size: 11px; color: #ccc;">Generazione</div>
                        </div>
                    </div>
                    """
                )

                workflow_box = gr.Markdown(
                        value="⏸️ In attesa di domanda...",
                        show_label=False
                    )
                with gr.Accordion("ℹ️ Come Funziona", open=False):    
                    gr.Markdown(
                            """
                            **Fasi del Workflow:**
                            1. 🔄 **Routing** - Classificazione tipo di query
                            2. 📚 **Retrieval** - Ricerca documenti e database (RAG/TOOLS)
                            3. 🤖 **Generazione** - Risposta finale
                            """
                        )
                with gr.Accordion("🧠 Reasoning", open=False):        
                    thinking_checkbox = gr.Checkbox(
                            label="🔄 Router Reasoning",
                            value=False,
                            info="Abilita la modalità di ragionamento del router (più lento ma più accurato)"
                        )
                
                with gr.Accordion("🔧 Strumenti Disponibili (25 Tools)", open=False):
                    gr.Markdown(
                        """
                        ### 🔥 Incendi - Fire Hotspots (12 tools)
                        
                        **Geographic Queries (3):**
                        - `get_hotspots_by_region` - Incendi per regione
                        - `get_hotspots_by_province` - Incendi per provincia
                        - `get_hotspots_by_municipality` - Incendi per comune
                        
                        **Attribute Filters (4):**
                        - `get_hotspots_by_intensity` - Filtra per intensità
                        - `get_hotspots_by_confidence` - Filtra per confidenza
                        - `get_hotspots_by_sensor` - Filtra per sensore (MODIS/VIIRS)
                        - `get_hotspots_by_satellite` - Filtra per satellite
                        
                        **Temporal Queries (2):**
                        - `get_hotspots_by_date` - Incendi per data
                        - `get_hotspots_by_time_of_day` - Incendi per periodo (Giorno/Notte)
                        
                        **Spatial Queries (2):**
                        - `get_hotspots_within_distance` - Incendi entro raggio
                        - `get_hotspots_in_bounding_box` - Incendi in area
                        
                        **Statistics (1):**
                        - `get_hotspots_statistics` - Panoramica completa
                        
                        ---
                        
                        ### 🌊 Alluvioni - Flood Bulletins (10 tools)
                        
                        **Geographic & Risk (4):**
                        - `get_flood_zones_by_region` - Allerte per regione
                        - `get_flood_zones_by_risk_level` - Filtra per livello (Verde/Giallo/Arancione/Rosso)
                        - `get_flood_zones_by_risk_class` - Filtra per classe numerica (0-3)
                        - `get_flood_zones_by_minimum_risk_class` - Zone con rischio >= N
                        
                        **Search & Pattern (2):**
                        - `get_flood_zones_by_zone_code` - Cerca per codice zona
                        - `get_flood_zones_with_name_pattern` - Cerca pattern nel nome
                        
                        **Spatial (2):**
                        - `get_flood_zones_within_distance` - Zone entro raggio
                        - `get_flood_zones_in_bounding_box` - Zone in area
                        
                        **Statistics & Comparison (2):**
                        - `get_flood_statistics` - Panoramica allerte
                        - `compare_flood_bulletins` - Confronto oggi/domani
                        
                        ---
                        
                        ### 🛠️ Utility (3 tools)
                        - `list_available_files` - Elenca file disponibili
                        - `get_available_regions` - Regioni con dati
                        - `get_vigilance_data_summary` - Riepilogo completo
                        """
                    )
                    

        # Examples section in collapsible accordion below the main row
        gr.Examples(
                examples=[
                    # RAG - Historical data from documents
                    "Quanta superficie totale è stata bruciata in Italia nel 2021?",
                    "Quale regione ha avuto la maggiore superficie forestale bruciata nel 2021?",
                    "Quanta superficie forestale è stata bruciata in Italia nel 2023?",
                    "Quanti comuni italiani erano a rischio per frane e alluvioni nel 2018?",
                    "Quanta popolazione era a rischio alluvioni in Italia nel 2018?",
                    "Qual è stata la variazione percentuale della pericolosità da frana tra 2018 e 2021?",
                    "Quanti abitanti erano a rischio frane nel 2021?",
                    "Descrivi l'incendio del Montiferru-Planargia in Sardegna nel 2021",
                    "Qual è l'impatto degli incendi sulle aree protette in Italia?",
                    # TOOL - Recent data queries
                    "Quanti incendi ci sono in Sicilia?",
                    "Mostrami gli incendi in Lombardia",
                    "Incendi nella provincia di Palermo",
                    "Ci sono incendi a Roma?",
                    "Mostrami gli incendi ad alta intensità",
                    "Incendi con alta confidenza",
                    "Dati del sensore VIIRS",
                    "Incendi nelle ultime 24 ore",
                    "Mostrami gli incendi di oggi",
                    "Incendi notturni",
                    "Ci sono incendi entro 50 km da Roma?",
                    "Dammi una panoramica generale degli incendi",
                    "Statistiche sugli incendi attuali",
                    "Allerte in Emilia-Romagna oggi",
                    "Quali zone hanno allerta arancione?",
                    "Zone con allerta rossa oggi",
                    "Allerte per domani in Toscana",
                    "Cerca la zona Lomb-A",
                    "Zone con 'Po' nel nome",
                    "Allerte entro 100 km da Bologna",
                    "Panoramica delle allerte oggi",
                    "Come cambia la situazione tra oggi e domani?",
                    "Confronta allerte oggi vs domani",
                    # DIRECT - General questions
                    "Che cos'è un'alluvione?",
                    "Spiega le cause principali degli incendi boschivi",
                    "Come si prevengono le alluvioni?",
                    "Quali sono i rischi idrogeologici in Italia?",
                    "Come funziona la prevenzione incendi?",
                    "Cosa sono i sensori MODIS e VIIRS?"
                ],
                inputs=query_input,
                label="📋 Esempi di Domande (clicca per provare)",
                elem_id="example-buttons",
                examples_per_page=7,
                example_labels=[
                    # RAG examples
                    "� RAG: Incendi 2021 - Superficie totale",
                    "� RAG: Incendi 2021 - Regioni colpite",
                    "� RAG: Incendi 2023 - Foreste",
                    "� RAG: Alluvioni 2018 - Comuni a rischio",
                    "� RAG: Alluvioni 2018 - Popolazione",
                    "📚 RAG: Frane 2021 - Variazione %",
                    "📚 RAG: Frane 2021 - Rischio",
                    "� RAG: Evento Montiferru",
                    "📚 RAG: Aree protette",
                    # TOOL examples - Fire Geographic
                    "🔥 TOOL: Incendi Sicilia",
                    "🔥 TOOL: Incendi Lombardia",
                    "🔥 TOOL: Incendi Palermo",
                    "🔥 TOOL: Incendi Roma",
                    # TOOL examples - Fire Attributes
                    "🔥 TOOL: Alta intensità",
                    "🔥 TOOL: Alta confidenza",
                    "🔥 TOOL: Sensore VIIRS",
                    # TOOL examples - Fire Temporal
                    "🔥 TOOL: Ultime 24 ore",
                    "🔥 TOOL: Incendi oggi",
                    "🔥 TOOL: Incendi notturni",
                    # TOOL examples - Fire Spatial
                    "🔥 TOOL: 50 km da Roma",
                    "🔥 TOOL: Panoramica incendi",
                    "� TOOL: Statistiche",
                    # TOOL examples - Flood Geographic & Risk
                    "🌊 TOOL: Allerte Emilia-Romagna",
                    "🌊 TOOL: Allerta arancione",
                    "🌊 TOOL: Allerta rossa",
                    "🌊 TOOL: Allerte domani Toscana",
                    # TOOL examples - Flood Search & Spatial
                    "🌊 TOOL: Zona Lomb-A",
                    "🌊 TOOL: Zone con 'Po'",
                    "🌊 TOOL: 100 km da Bologna",
                    # TOOL examples - Flood Stats & Comparison
                    "🌊 TOOL: Panoramica allerte",
                    "🌊 TOOL: Situazione oggi/domani",
                    "🌊 TOOL: Confronto allerte",
                    # DIRECT examples
                    "💡 DIRECT: Cos'è un'alluvione?",
                    "💡 DIRECT: Cause incendi",
                    "💡 DIRECT: Prevenzione alluvioni",
                    "💡 DIRECT: Rischi idrogeologici",
                    "💡 DIRECT: Prevenzione incendi",
                    "💡 DIRECT: Sensori MODIS/VIIRS"
                ]
            )
                

        
        # Event handlers
        def respond(message, chat_history, workflow_status, enable_thinking):
            # Process query and get workflow updates
            response, workflow_update, pipeline_html = process_query(message, chat_history, workflow_status, enable_thinking)
            # Use messages format for chatbot
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": response})
            return "", chat_history, workflow_update, pipeline_html
        
        submit_btn.click(
            respond,
            inputs=[query_input, chatbot, workflow_box, thinking_checkbox],
            outputs=[query_input, chatbot, workflow_box, pipeline_status]
        )
        
        query_input.submit(
            respond,
            inputs=[query_input, chatbot, workflow_box, thinking_checkbox],
            outputs=[query_input, chatbot, workflow_box, pipeline_status]
        )
        
        clear_btn.click(
            lambda: (None, "Chat pulita", "⏸️ In attesa di domanda...", get_pipeline_html("idle")),
            outputs=[chatbot, query_input, workflow_box, pipeline_status]
        )
    
    return demo


def main():
    """Main entry point"""
    global agent
    
    port = int(os.getenv("GRADIO_SERVER_PORT", 7860))
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    
    logger.info(f"Starting Elementis on port {port}")
    
    # Initialize agent before launching UI to avoid double initialization
    logger.info("Pre-initializing agent...")
    try:
        agent = Agent()
        logger.info("Agent pre-initialized successfully")
    except Exception as e:
        logger.error(f"Failed to pre-initialize agent: {e}")
    
    demo = create_interface()
    demo.launch(
        server_port=port,
        share=share,
        server_name="127.0.0.1"  # Use localhost for Windows compatibility
    )


if __name__ == "__main__":
    main()
