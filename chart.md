## Agent-Mediated Service Abstraction with MCP
[Pattern description]
### Data Flow Diagram
```mermaid
sequenceDiagram
    participant U as User
    participant UI as Web Application UI
    participant AO as Agent Orchestrator
    participant LLM as LLM
    participant TW as Tool Implementation Wrappers
    participant MC as MCP Client
    participant MS as MCP Server
    participant BS as Backend Services/APIs

    U->>UI: Query (e.g., "Price of AAPL?")
    UI->>AO: Forward Query
    AO->>LLM: Send Query, History, Tool Definitions
    LLM-->>AO: Select Tool (e.g., get_price)
    AO->>TW: Invoke Tool Wrapper
    TW->>MC: Call MCP Tool
    MC->>MS: Standardized Request
    MS->>BS: Authenticated API Call
    BS-->>MS: API Response
    MS-->>MC: Processed Data
    MC-->>TW: Return Data
    TW-->>AO: Formatted Result
    AO->>LLM: Send Tool Result
    LLM-->>AO: Synthesize Response
    AO-->>UI: Conversational Response
    UI-->>U: Display Response
