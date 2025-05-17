# 🧠 Medical Research Agent with Smithery.ai, MCP & Fetch.ai uAgents

[![uAgents](https://img.shields.io/badge/uAgents-Framework-blue)](https://github.com/fetchai/uAgents)
[![MCP](https://img.shields.io/badge/MCP-Protocol-green)](https://github.com/modelcontextprotocol)
[![Smithery.ai](https://img.shields.io/badge/Smithery.ai-MCP%20Servers-orange)](https://smithery.ai)

This project demonstrates how to build a chat-enabled medical research agent using:

- 🔌 **Model Context Protocol (MCP)**
- 🌐 **Smithery.ai MCP Server Network**
- 🤖 **Fetch.ai uAgents Framework**
- 💬 **ASI:One and Agentverse** for real-time interaction

The agent uses Anthropic Claude to decide which tool to invoke, then connects to MCP servers (e.g., PubMed, Clinical Trials, Calculators) via streamable HTTP and returns structured results to the user.

## ✨ Features

- 🔍 **Query PubMed**, medical calculators, clinical trials, and more
- 🔌 **Connects to multiple Smithery-hosted MCP servers** via standard protocols
- 🤝 **Powered by Fetch.ai uAgents** with secure messaging and identities
- 🧠 **Tool routing and response formatting** handled by Claude 3.5 Sonnet
- 💬 **Supports ASI:One chat protocol** for real-time LLM interaction

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/kshipra-fetch/smitheryai-mcp-integration.git
cd smitheryai-mcp-integration
```

### 2. Install Dependencies

Create a virtual environment and install the packages:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


### 3. Set Environment Variables

Create a `.env` file in the root directory with the following:

```env
SMITHERY_API_KEY=your_smithery_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

## 🧪 MCP Servers Used

The agent connects to 5 MCP servers hosted on Smithery.ai:

| Server Path | Description |
|-------------|-------------|
| `@vitaldb/medcalc` | Performs medical calculations like BMI, HOMA-IR |
| `@JackKuo666/clinicaltrials-mcp-server` | Accesses clinical trial databases |
| `@JackKuo666/pubmed-mcp-server` | Queries biomedical literature on PubMed |
| `@openags/paper-search-mcp` | Searches scientific research metadata |
|`@nickclyde/duckduckgo-mcp-server`|Searches the web for information|

## 💡 How It Works

1. ✅ The agent initializes with the Chat Protocol and connects to MCP servers
2. 💬 A user sends a message (e.g., "Find clinical trials for Alzheimer's")
3. 🧠 Claude decides which tool to invoke and constructs the input schema
4. 🌐 The agent sends the request to the appropriate MCP server via `streamablehttp_client`
5. 🧾 The response is returned, formatted by Claude for clarity, and sent back to the user

## 🗨️ Chat Protocol Support

The agent implements:

- **ChatMessage**: Receive user messages
- **ChatAcknowledgement**: Confirm receipt

This makes the agent fully compatible with ASI:One and Agentverse for real-time messaging.

## 📎 Example Usage

```bash
python agent.py
```

After you run the agent, an agent inspector link will appear in the terminal 
```bash
ValueError: Please set the ANTHROPIC_API_KEY environment variable in your .env file
(venv) (base) kshipra@MacBook-Pro-3 smithery-ai % python3 smithery-ai.py
INFO:     [MedicalResearchMCPAgent]: Starting agent with address: agent1qgfla63yp9g6nsv5xhzdr2c43qqrehw9wq5d39y3lcflhuanpsmfk0gqcql
INFO:     [MedicalResearchMCPAgent]: Agent inspector available at https://agentverse.ai/inspect/?uri=http%3A//127.0.0.1%3A8001&address=agent1qgfla63yp9g6nsv5xhzdr2c43qqrehw9wq5d39y3lcflhuanpsmfk0gqcql
INFO:     [MedicalResearchMCPAgent]: Starting server on http://0.0.0.0:8001 (Press CTRL+C to quit)
INFO:     [MedicalResearchMCPAgent]: Starting mailbox client for https://agentverse.ai
INFO:     [MedicalResearchMCPAgent]: Mailbox access token acquired
INFO:     [uagents.registration]: Registration on Almanac API successful
WARNING:  [uagents.registration]: Mismatch in almanac contract versions: supported (2.1.0), deployed (2.3.0). Update uAgents to the latest version for compatibility.
INFO:     [uagents.registration]: Registering on almanac contract...
```

Open the link "https://agentverse.ai/inspect/?uri=http%3A//127.0.0.1%3A8001&address=agent1qgfla63yp9g6nsv5xhzdr2c43qqrehw9wq5d39y3lcflhuanpsmfk0gqcql" in a browser and click "Connect". This will connect your Agent to Agentverse and you can start interacting with it by clicking on Agent Profile, switch to Overview and select "Chat with Agent" alternatively you can also query your agent through [ASI:One](https://asi1.ai/)


### Example Queries

- "Find recent clinical trials for diabetes treatment"
- "Calculate BMI for a 70kg person who is 175cm tall"
- "Search PubMed for papers on machine learning in oncology"
- "What are the latest research papers on cardiovascular disease?"


## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
