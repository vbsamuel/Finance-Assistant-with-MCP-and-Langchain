
# Conversational Finance Assistant (FastMCP + Langchain + Streamlit)

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) <!-- Choose a license! MIT is common -->

This project demonstrates building a conversational financial assistant capable of retrieving real-time stock quotes, news headlines, and market mover data using natural language queries.

It features a decoupled architecture:
*   **Backend:** A secure server built with **FastMCP** acting as a gateway to financial APIs (Finnhub, Alpha Vantage).
*   **Frontend:** A user-friendly web interface built with **Streamlit**.
*   **Agent:** Powered by **Langchain** and the **OpenAI API** (GPT models) to understand user requests, utilize backend tools, and generate conversational responses.

**Technical Report:**

(Report)[https://github.com/Pyligent/Finance-Assistant-with-MCP-and-Langchain/blob/main/report.md]

## Features

*   **Natural Language Queries:** Ask questions like:
    *   "What's the price of Apple?"
    *   "How is MSFT doing today?"
    *   "Any recent news for TSLA?"
    *   "Show me the top gainers today."
    *   "What's the market news?"
*   **Real-time Data:** Fetches current stock quotes, recent news, and market movers via external APIs.
*   **Secure API Key Management:** Financial API keys are stored securely on the backend MCP server, not exposed in the frontend or to the LLM.
*   **Conversational Responses:** The LLM synthesizes data fetched via tools into easy-to-understand answers.
*   **Follow-up Suggestions:** Provides relevant next questions to continue the conversation.
*   **Modular Architecture:** Decouples the UI/Agent logic from the backend data fetching logic using the Model Context Protocol (MCP).

## Architecture

The system uses a client-server architecture orchestrated by a Langchain agent:

1.  **User Interface (Streamlit):** Handles chat display, user input, and suggestion buttons.
2.  **Langchain Agent Executor:** Resides in the Streamlit app. Uses `ChatOpenAI` and defined `StructuredTool`s. Manages the conversation flow, calls the LLM, and executes tools when requested.
3.  **OpenAI LLM:** Interprets user intent, decides when to call tools, synthesizes final responses from tool results.
4.  **Langchain Tools (in UI):** Python functions (`get_price`, `get_news`, `get_market_movers`) defined within the UI code. These tools are invoked by the Agent Executor.
5.  **FastMCP Client (in UI Tools):** The Langchain tools use `fastmcp.Client` to communicate with the backend MCP server.
6.  **FastMCP Server (Backend):** A separate Python process (`fin_server_v2.py`). Exposes financial data fetching capabilities as secure MCP Tools (`@mcp.tool()`) and Resources (`@mcp.resource()`). Handles interaction with external financial APIs.
7.  **Financial APIs:** Finnhub and Alpha Vantage (can be extended).


## Setup and Installation

**Prerequisites:**

*   Python 3.10+
*   `uv` (recommended) or `pip`
*   API Keys:
    *   OpenAI API Key ([platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys))
    *   Finnhub API Key ([finnhub.io](https://finnhub.io/))
    *   Alpha Vantage API Key ([alphavantage.co](https://www.alphavantage.co/))

**Steps:**

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create `.env` File:**
    Create a file named `.env` in the project root and add your API keys:
    ```dotenv
    # .env file
    FINNHUB_API_KEY=YOUR_FINNHUB_KEY
    ALPHA_VANTAGE_API_KEY=YOUR_ALPHA_VANTAGE_KEY
    OPENAI_API_KEY=sk-YOUR_OPENAI_KEY
    ```
    *(Replace the placeholder values with your actual keys)*

3.  **Create Virtual Environment:**
    ```bash
    uv venv # Creates a .venv folder
    source .venv/bin/activate # On Linux/macOS
    # .\venv\Scripts\activate # On Windows CMD/PowerShell
    ```

4.  **Install Dependencies:**
    ```bash
    uv pip install -r requirements.txt
    # OR if you don't have a requirements.txt yet:
    # uv pip install streamlit "fastmcp" httpx python-dotenv pydantic-settings openai langchain langchain-openai pydantic langchainhub "langchain-community"
    ```
    *(See `requirements.txt` for specific tested versions)*

## Running the Application

You need to run the backend MCP server and the frontend Streamlit UI separately.

1.  **Run the Backend MCP Server:**
    Open a terminal, activate the virtual environment, and run:
    ```bash
    python fin_server_v2.py
    ```
    Keep this terminal window open. You should see log messages indicating it started successfully and loaded API keys.

2.  **Run the Frontend Streamlit UI:**
    Open a *second* terminal window, activate the *same* virtual environment, and run:
    ```bash
    streamlit run fin_langchain_v2.py
    ```
    Streamlit will provide a local URL (usually `http://localhost:8501`). Open this URL in your web browser.

3.  **Interact:** Start asking financial questions in the chat interface!

## Code Structure

*   `fin_server_v2.py`: The backend FastMCP server application. Contains tool and resource definitions, interacts with financial APIs.
*   `fin_langchain_v2.py`: The frontend Streamlit application. Contains the Langchain agent setup, UI components, and helper functions to call the MCP server.
*   `.env` (You create this): Stores API keys securely.
*   `requirements.txt` (You create this or use the one provided): Lists Python dependencies.

## Future Improvements

*   Add more financial tools (historical data, fundamentals, analyst ratings).
*   Implement more sophisticated error handling and API fallback logic.
*   Improve NLU for ticker/company name recognition.
*   Integrate Langchain memory more deeply for multi-turn context.
*   Add data visualization (charts) to the Streamlit UI.
*   Implement server-side caching for financial APIs.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

## License

This project is licensed under the [MIT License](LICENSE). <!-- Choose and add a LICENSE file -->

