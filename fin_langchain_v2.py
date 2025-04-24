# finance_assistant_ui.py
import streamlit as st
import asyncio
import json
import re
import inspect
from typing import List, Dict, Any, Optional, Tuple, Type, Union

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# --- Ensure OPENAI_API_KEY is set ---
import os
if not os.getenv("OPENAI_API_KEY"):
    st.error("FATAL: OPENAI_API_KEY environment variable not set. Langchain/OpenAI requires this.")
    st.stop()
# --- ---

# FastMCP Client
from fastmcp import Client
from fastmcp.exceptions import ClientError
from mcp.types import TextContent, TextResourceContents

# Langchain Components
try:
    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain.memory import ConversationBufferWindowMemory # Using windowed memory
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain.tools import StructuredTool # Use StructuredTool for pydantic args
    from langchain import hub
    from langchain_core.exceptions import OutputParserException # For handling agent errors
except ImportError as e:
    st.error(f"ImportError: Could not import Langchain components: {e}. Please ensure 'langchain', 'langchain-openai', 'pydantic', and 'langchainhub' are installed correctly.")
    st.stop()

# OpenAI Client (needed for separate suggestion call)
try:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletionMessageToolCall # Although agent handles calls, type hint might be useful
except ImportError as e:
    st.error(f"ImportError: Could not import OpenAI components: {e}. Please ensure 'openai>=1.0.0' is installed correctly.")
    st.stop()


# Pydantic for Tool Arguments
from pydantic import BaseModel, Field, ValidationError
import traceback
import logging

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_ui = logging.getLogger("FinanceUI_Langchain_Agent_Suggest")
# --- ---

# --- Configuration ---
MCP_SERVER_TARGET = "fin_server_v2.py" # Make sure this points to your corrected server file
OPENAI_MODEL = "gpt-4-turbo-preview"
AGENT_PROMPT_HUB_REPO = "hwchase17/openai-tools-agent"
MEMORY_K = 5 # Number of past interactions for the agent to remember
# --- ---

# --- MCP Client Helper Functions ---
async def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to call an MCP tool asynchronously. Returns result dict or raises Exception on MCP/connection error."""
    log_ui.info(f"Agent -> MCP Call: {tool_name} with args: {arguments}")
    try:
        async with Client(MCP_SERVER_TARGET) as client:
            result = await client.call_tool(tool_name, arguments, _return_raw_result=True)
            if result.isError:
                error_text = result.content[0].text if result.content and isinstance(result.content[0], TextContent) else "Unknown tool error"
                log_ui.error(f"MCP Tool Error ({tool_name}): {error_text}")
                raise Exception(f"Error from MCP tool '{tool_name}': {error_text}")
            elif not result.content:
                 log_ui.warning(f"MCP Tool ({tool_name}) returned no content.")
                 return {"mcp_result": "Tool executed successfully but returned no specific data."}
            else:
                if isinstance(result.content[0], TextContent):
                    try:
                        parsed_content = json.loads(result.content[0].text)
                        log_ui.info(f"MCP Tool ({tool_name}) successful, parsed JSON.")
                        return parsed_content
                    except (json.JSONDecodeError):
                         log_ui.info(f"MCP Tool ({tool_name}) successful, returned raw text.")
                         return {"mcp_result": result.content[0].text}
                else:
                    log_ui.info(f"MCP Tool ({tool_name}) successful, returned non-text content: {type(result.content[0])}")
                    return {"mcp_result": f"Received non-text content: {type(result.content[0])}"}
    except (ClientError, ConnectionRefusedError, FileNotFoundError) as e:
        log_ui.error(f"MCP Client/Connection Error calling {tool_name}: {e}", exc_info=False)
        st.error(f"Failed to connect to or call MCP server for {tool_name}: {e}")
        raise Exception(f"Failed to reach MCP server for tool {tool_name}: {e}") from e
    except Exception as e:
        log_ui.error(f"Unexpected error calling MCP tool {tool_name}: {e}", exc_info=True)
        st.error(f"Unexpected error calling MCP tool {tool_name}: {e}")
        raise Exception(f"Unexpected Error during MCP tool call {tool_name}: {e}") from e

async def read_mcp_resource(uri: str) -> List[Dict[str, Any]] | Dict[str, Any]:
    """Helper to read an MCP resource asynchronously. Returns parsed JSON list/dict or raises Exception on error."""
    log_ui.info(f"Agent -> MCP Read: {uri}")
    try:
        async with Client(MCP_SERVER_TARGET) as client:
            result = await client.read_resource(uri)
            if not result:
                 log_ui.warning(f"MCP Resource ({uri}) returned no content.")
                 return {"mcp_result": "Resource exists but is empty."}
            else:
                if len(result) > 0 and hasattr(result[0], 'text') and isinstance(result[0].text, str):
                    try:
                        parsed_content = json.loads(result[0].text)
                        log_ui.info(f"MCP Resource ({uri}) successful, parsed JSON.")
                        return parsed_content
                    except (json.JSONDecodeError):
                         log_ui.info(f"MCP Resource ({uri}) successful, returned text.")
                         return {"mcp_result": result[0].text}
                elif len(result) > 0 and hasattr(result[0], 'blob'):
                     log_ui.info(f"MCP Resource ({uri}) successful, returned binary data.")
                     return {"mcp_result": f"Received binary data (length: {len(result[0].blob)})"}
                else:
                     log_ui.warning(f"MCP Resource ({uri}) returned unexpected content type: {type(result[0]) if result else 'None'}")
                     return {"mcp_result": f"Received unknown content type: {type(result[0]) if result else 'None'}"}
    except (ClientError, ConnectionRefusedError, FileNotFoundError) as e:
        log_ui.error(f"MCP Client/Connection Error reading {uri}: {e}", exc_info=False)
        st.error(f"Failed to connect to or read MCP resource {uri}: {e}")
        raise Exception(f"Failed to reach MCP server for resource {uri}: {e}") from e
    except Exception as e:
        log_ui.error(f"Unexpected error reading MCP resource {uri}: {e}", exc_info=True)
        st.error(f"Unexpected error reading MCP resource {uri}: {e}")
        raise Exception(f"Unexpected Error reading MCP resource {uri}: {e}") from e

# --- Pydantic Models for Langchain Tool Arguments ---
class GetPriceSchema(BaseModel):
    ticker: str = Field(..., description="The stock ticker symbol (e.g., AAPL, MSFT, GOOGL).")

class GetNewsSchema(BaseModel):
    ticker: Optional[str] = Field(None, description="The stock ticker symbol (e.g., AAPL, MSFT). Omit for general market news.")

class GetMarketMoversSchema(BaseModel):
     limit_per_category: int = Field(5, description="Maximum number of stocks per category (gainers, losers, active). Defaults to 5.")

# --- Core Logic Functions (Called by Langchain Tools) ---
# These return the raw data dict/list or raise exceptions
async def _get_price_logic(ticker: str) -> dict:
    """Core logic: Gets stock price quote dict from MCP server."""
    log_ui.info(f"Executing tool logic 'get_price' for ticker: {ticker}")
    return await call_mcp_tool("get_stock_price", {"ticker": ticker})

async def _get_news_logic(ticker: Optional[str] = None) -> dict | list:
    """Core logic: Gets news list/dict from MCP server."""
    log_ui.info(f"Executing tool logic 'get_news' for ticker: {ticker}")
    if ticker:
        uri = f"news://ticker/{ticker.upper()}"
        return await read_mcp_resource(uri)
    else:
        uri = "news://market/general"
        return await read_mcp_resource(uri)

async def _get_market_movers_logic(limit_per_category: int = 5) -> dict:
    """Core logic: Gets market movers dict from MCP server."""
    log_ui.info(f"Executing tool logic 'get_market_movers' with limit: {limit_per_category}")
    return await call_mcp_tool("get_market_movers", {"limit_per_category": limit_per_category})


# --- Create Langchain StructuredTool Definitions ---
# Tools now directly use the logic functions
tools_list = [
    StructuredTool.from_function(
        func=None, coroutine=_get_price_logic, name="get_price",
        description="Gets the current stock price quote for a given ticker symbol (e.g., AAPL, MSFT, GOOGL). Always provide the ticker symbol.",
        args_schema=GetPriceSchema, return_direct=False, handle_tool_error=True
    ),
    StructuredTool.from_function(
        func=None, coroutine=_get_news_logic, name="get_news",
        description="Gets recent news headlines. Provide a ticker symbol (e.g., AAPL, MSFT) for company-specific news, or omit the ticker for general market news.",
        args_schema=GetNewsSchema, return_direct=False, handle_tool_error=True
    ),
    StructuredTool.from_function(
        func=None, coroutine=_get_market_movers_logic, name="get_market_movers",
        description="Gets lists of the current top gaining, top losing, and most actively traded stocks for the US market. Use this when asked about 'top stocks', 'biggest winners', 'biggest losers', 'market movers', 'most active stocks', etc.",
        args_schema=GetMarketMoversSchema, return_direct=False, handle_tool_error=True
    )
]

# --- Function to Generate Suggestions ---
async def generate_suggestions(history: List[Dict[str, str]]) -> List[str]:
    """Generates follow-up suggestions using OpenAI."""
    if not history or len(history) < 2: # Need at least user query and assistant response
        log_ui.info("Not enough history to generate suggestions.")
        return []

    # Use only the last user message and last assistant response for context
    last_user = next((msg for msg in reversed(history) if msg["role"] == "user"), None)
    last_assistant = next((msg for msg in reversed(history) if msg["role"] == "assistant"), None)

    if not (last_user and last_assistant):
        log_ui.info("Could not find last user/assistant pair for suggestions.")
        return []

    openai_client = AsyncOpenAI()
    suggestions = []
    suggestion_prompt_context = [last_user, last_assistant]

    try:
        log_ui.info("Generating follow-up suggestions...")
        suggestion_request_messages = [
            {"role": "system", "content": "Based on the last user query and assistant response, suggest exactly 3 distinct, relevant follow-up questions the user might ask next. Output *only* a JSON list of strings, like `[\"Question 1?\", \"Question 2?\", \"Question 3?\"]`. Do not include any other text or explanation."},
        ] + suggestion_prompt_context

        suggestion_response = await openai_client.chat.completions.create(
            model=OPENAI_MODEL, messages=suggestion_request_messages, # type: ignore
            temperature=0.6, max_tokens=150, response_format={"type": "json_object"}
        )
        suggestion_content = suggestion_response.choices[0].message.content
        if suggestion_content:
            try:
                parsed_suggestions = json.loads(suggestion_content)
                # Handle potential wrapping like {"suggestions": [...]}
                if isinstance(parsed_suggestions, dict):
                     potential_list = next(iter(parsed_suggestions.values()), None)
                     if isinstance(potential_list, list): parsed_suggestions = potential_list
                if isinstance(parsed_suggestions, list) and all(isinstance(s, str) for s in parsed_suggestions):
                    suggestions = [s for s in parsed_suggestions if s][:3] # Filter empty strings and take top 3
                    log_ui.info(f"Generated suggestions: {suggestions}")
                else: log_ui.warning(f"LLM suggestion response was not a valid list of strings: {suggestion_content}")
            except json.JSONDecodeError: log_ui.warning(f"Failed to parse suggestions JSON: {suggestion_content}")
    except Exception as sugg_err:
        log_ui.error(f"Error generating suggestions: {sugg_err}", exc_info=True)
    return suggestions


# --- Formatting Function (Simplified for Display) ---
def format_display_response(text: str) -> str:
    """Passes through the LLM response, assuming it's pre-formatted markdown."""
    # Agent's final response should be formatted markdown based on system prompt
    return text

# --- Streamlit App UI ---

st.set_page_config(page_title="Finance Assistant", layout="wide")
st.title("📈 Conversational Finance Assistant (Langchain Agent)")
st.caption("Ask about stock prices, news, or market movers (e.g., 'Top gainers today?', 'Price of AAPL?', 'News for MSFT?').")

# Initialize chat history, memory, agent executor, and suggestions
if "messages" not in st.session_state:
    st.session_state.messages = [] # For display: List[Dict{"role": str, "content": str}]
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=MEMORY_K, memory_key="chat_history", return_messages=True, output_key="output"
    )
if "agent_executor" not in st.session_state:
    try:
        log_ui.info("Initializing Langchain Agent Executor...")
        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1)
        try:
            prompt_template = hub.pull(AGENT_PROMPT_HUB_REPO)
        except Exception as hub_err:
            log_ui.error(f"Could not pull prompt from Langchain Hub '{AGENT_PROMPT_HUB_REPO}': {hub_err}. Using basic fallback.")
            prompt_template = ChatPromptTemplate.from_messages([
                 ("system", "You are a helpful assistant."), MessagesPlaceholder(variable_name="chat_history"),
                 ("human", "{input}"), MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])

        # Define the refined system message content
        system_message_content = (
            "You are a helpful and conversational financial data assistant.\n"
            "**Available Tools:**\n"
            "- `get_price(ticker)`: Gets the CURRENT stock price quote for ONE specific ticker symbol.\n"
            "- `get_news(ticker=None)`: Gets RECENT news headlines for ONE specific ticker OR general market news if no ticker is given.\n"
            "- `get_market_movers(limit_per_category=5)`: Gets lists of CURRENT top gaining, losing, and most active US stocks.\n\n"
            "**Your Limitations:** You CANNOT give investment advice, recommendations, predictions, analyze fundamentals (P/E, etc.), access historical data, or explain *why* stocks moved.\n\n"
            "**How to Respond:**\n"
            "1. Understand the request (price, news, movers?). Infer tickers if possible (Google->GOOGL), ask if unsure.\n"
            "2. If the request is **outside your capabilities**, **DO NOT use a tool**. Politely state limitations and suggest what you *can* do (e.g., 'I can't give recommendations, but would you like the current price of...?').\n"
            "3. If a tool fits, call it.\n"
            "4. **Synthesize Results:** Convert the tool's JSON/dict output into a conversational, formatted response:\n"
            "   - *Price:* Use markdown bullets for Price, Change, %, High/Low, Source.\n"
            "   - *News:* Use numbered markdown list for headlines.\n"
            "   - *Movers:* Use markdown lists for top gainers/losers/active (Ticker and Change %).\n"
            "5. **Handle Tool Errors:** If the tool result contains `{'error': '...'}`, clearly state the error message to the user."
        )
        # Replace or prepend system message
        if prompt_template.messages and isinstance(prompt_template.messages[0], SystemMessage):
             prompt_template.messages[0].content = system_message_content
        else:
             prompt_template.messages.insert(0, SystemMessage(content=system_message_content))

        # Ensure necessary placeholders exist
        if "chat_history" not in prompt_template.input_variables: prompt_template.messages.insert(1, MessagesPlaceholder(variable_name="chat_history"))
        if "agent_scratchpad" not in prompt_template.input_variables: prompt_template.messages.append(MessagesPlaceholder(variable_name="agent_scratchpad"))

        agent = create_openai_tools_agent(llm, tools_list, prompt_template)
        st.session_state.agent_executor = AgentExecutor(
            agent=agent, tools=tools_list, memory=st.session_state.memory,
            verbose=True, handle_parsing_errors=True, max_iterations=5
        )
        log_ui.info("Langchain Agent Executor initialized successfully.")
    except Exception as e:
        st.error(f"Failed to initialize Langchain Agent: {e}")
        log_ui.error(f"Error initializing Langchain Agent: {traceback.format_exc()}")
        st.session_state.agent_executor = None
if "current_suggestions" not in st.session_state:
    st.session_state.current_suggestions = []

# Function to handle suggestion button click
def handle_suggestion_click(suggestion_text):
    log_ui.info(f"Suggestion clicked: '{suggestion_text}'")
    # Append suggestion as the new user prompt
    st.session_state.messages.append({"role": "user", "content": suggestion_text})
    # Clear suggestions so they disappear after clicking
    st.session_state.current_suggestions = []
    # Rerun to process the new message
    st.rerun()

# Display chat messages from Streamlit's history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(format_display_response(message["content"]))

# Display Suggestion Buttons
if st.session_state.current_suggestions:
    st.markdown("---")
    num_suggestions = len(st.session_state.current_suggestions)
    if num_suggestions > 0:
        # Dynamically create columns based on number of suggestions
        cols = st.columns(num_suggestions)
        for i, suggestion in enumerate(st.session_state.current_suggestions):
            # Basic check for non-empty suggestion
            if suggestion and isinstance(suggestion, str) and suggestion.strip():
                with cols[i]:
                    st.button(
                        suggestion,
                        key=f"suggestion_{i}", # Keys need to be stable within a render
                        on_click=handle_suggestion_click,
                        args=(suggestion,),
                        use_container_width=True
                     )
            else:
                log_ui.warning(f"Skipping empty or invalid suggestion at index {i}: {suggestion}")

# React to user input from chat box
latest_message_role = st.session_state.messages[-1]["role"] if st.session_state.messages else None

if prompt := st.chat_input("What financial info do you need?"):
    log_ui.info(f"New prompt received from chat input: '{prompt}'")
    st.session_state.current_suggestions = [] # Clear suggestions on new input
    st.session_state.messages.append({"role": "user", "content": prompt})
    latest_message_role = "user" # Ensure we process this new message
    st.rerun() # Display user message immediately

# Process the latest user message if it's from the user and agent is ready
if latest_message_role == "user" and st.session_state.agent_executor:
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response_content = ""
        generated_suggestions = []
        with st.spinner("Thinking..."):
             try:
                log_ui.info("Invoking Langchain agent executor...")
                # AgentExecutor handles the tool calling loop internally
                response = asyncio.run(st.session_state.agent_executor.ainvoke(
                    {"input": st.session_state.messages[-1]["content"]}
                ))
                full_response_content = response.get('output', "Sorry, I didn't get a valid response.")
                log_ui.info(f"Agent execution finished. Output received.")

                # --- Generate suggestions based on the final interaction ---
                # Use the *display* history (simple dicts) for context
                if st.session_state.messages:
                     history_for_suggestions = st.session_state.messages[-2:] # Max last user/assistant pair
                     generated_suggestions = asyncio.run(generate_suggestions(history_for_suggestions))
                else:
                     generated_suggestions = []

             except OutputParserException as parse_err:
                 log_ui.error(f"Langchain Output Parsing Error: {parse_err}", exc_info=True)
                 st.error(f"Error understanding the response structure: {parse_err}")
                 full_response_content = f"Sorry, there was an issue processing the response. ({parse_err})"
                 generated_suggestions = []
             except Exception as e:
                log_ui.error(f"Error during agent execution or suggestion generation: {e}", exc_info=True)
                st.error(f"Error processing request: {e}")
                st.error(f"Traceback: {traceback.format_exc()}")
                full_response_content = "Sorry, I encountered an internal error processing your request."
                generated_suggestions = []

        # Use the simplified formatter for the final display
        formatted_response = format_display_response(full_response_content)
        message_placeholder.markdown(formatted_response)

    # Add assistant response to Streamlit display history
    st.session_state.messages.append({"role": "assistant", "content": full_response_content})
    # Store suggestions for the next render cycle
    st.session_state.current_suggestions = generated_suggestions
    st.rerun() # Rerun to display the new assistant message and suggestions

elif latest_message_role == "user" and not st.session_state.agent_executor:
     # Handle case where agent failed to initialize
     st.error("Agent executor not initialized. Cannot process request.")
     # Avoid infinite loop by adding a response here
     if st.session_state.messages[-1]["role"] == "user": # Check to prevent duplicate errors
          st.session_state.messages.append({"role": "assistant", "content": "Error: Assistant agent not initialized."})
          st.rerun()