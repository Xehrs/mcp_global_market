from together import Together
from dotenv import load_dotenv
from uagents_core.contrib.protocols.chat import (
    chat_protocol_spec,
    ChatMessage,
    ChatAcknowledgement,
    TextContent,
    StartSessionContent,
)
from uagents import Agent, Context, Protocol
from uagents.setup import fund_agent_if_low
from datetime import datetime, timezone, timedelta
from uuid import uuid4
import mcp
from mcp.client.streamable_http import streamablehttp_client
import json
import base64
import asyncio
from typing import Dict, List, Optional, Any
from contextlib import AsyncExitStack
import os

# Load environment variables
load_dotenv()

# Get Together AI API key from environment variable
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise ValueError("Please set the TOGETHER_API_KEY environment variable in your .env file")

SMITHERY_API_KEY = os.getenv("SMITHERY_API_KEY")
if not SMITHERY_API_KEY:
    raise ValueError("Please set the SMITHERY_API_KEY environment variable in your .env file")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable in your .env file")

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
if not BRAVE_API_KEY:
    raise ValueError("Please set the BRAVE_API_KEY environment variable in your .env file")

class AnalysisResearchMCPClient:
    def __init__(self):
        self.sessions: Dict[str, mcp.ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self.together = Together(api_key=TOGETHER_API_KEY)
        self.all_tools = []
        self.tool_server_map = {}
        self.server_configs = {}
        self.default_timeout = timedelta(seconds=30)  # Default timeout of 30 seconds

    def get_server_config(self, server_path: str) -> dict:
        """Get or create server configuration"""
        if server_path not in self.server_configs:
            config_templates = {
                "@smithery-ai/brave-search": {
                    "braveApiKey": BRAVE_API_KEY
                },
                "@esxr/contract_manufacturers_mcp": {
                    "OPENAI_API_KEY": OPENAI_API_KEY
                },
            }
            self.server_configs[server_path] = config_templates.get(server_path, {})
        return self.server_configs[server_path]

    async def connect_to_servers(self, ctx: Context):
        """Connect to all MCP servers and collect their tools"""
        base_config = {
            "ignoreRobotsTxt": True
        }


        servers = [
            "@smithery-ai/brave-search",
            "@esxr/contract_manufacturers_mcp",
        ]

        for server_path in servers:
            try:
                ctx.logger.info(f"Connecting to server: {server_path}")
                server_config = self.get_server_config(server_path)
                config = {**base_config, **server_config}
                config_b64 = base64.b64encode(json.dumps(config).encode()).decode()
                url = f"https://server.smithery.ai/{server_path}/mcp?config={config_b64}&api_key={SMITHERY_API_KEY}"

                try:
                    read_stream, write_stream, _ = await self.exit_stack.enter_async_context(
                        streamablehttp_client(url)
                    )
                    session = await self.exit_stack.enter_async_context(
                        mcp.ClientSession(read_stream, write_stream)
                    )

                    await session.initialize()

                    tools_result = await session.list_tools()
                    tools = tools_result.tools

                    self.sessions[server_path] = session
                    for tool in tools:
                        tool_info = {
                            "name": tool.name,
                            "description": f"[{server_path}] {tool.description}",
                            "input_schema": tool.inputSchema,
                            "server": server_path,
                            "tool_name": tool.name
                        }
                        self.all_tools.append(tool_info)
                        self.tool_server_map[tool.name] = server_path

                        print(f"{tool_info["name"]}: {tool_info["input_schema"]}" )

                    ctx.logger.info(f"Successfully connected to {server_path}")
                    ctx.logger.info(f"Available tools: {', '.join([t.name for t in tools])}")

                except Exception as e:
                    ctx.logger.error(f"Error during connection setup: {str(e)}")
                    ctx.logger.error(f"Error type: {type(e)}")
                    import traceback
                    ctx.logger.error(f"Traceback: {traceback.format_exc()}")
                    raise

            except Exception as e:
                ctx.logger.error(f"Error connecting to {server_path}: {str(e)}")
                ctx.logger.error(f"Error type: {type(e)}")
                import traceback
                ctx.logger.error(f"Traceback: {traceback.format_exc()}")
                continue

    async def process_query(self, query: str, ctx: Context) -> str:
        try:
            messages = [{"role": "user", "content": query}]
            model_tools = [{
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            } for tool in self.all_tools]
            loop_actions = []
            max_loops = 8
            for _ in range(max_loops):
                response = self.together.chat.completions.create(
                    model="deepseek-ai/DeepSeek-V3",
                    max_tokens=1000,
                    messages=messages,
                    tools=model_tools
                )
                if not response.choices or not response.choices[0].message:
                    break
                message = response.choices[0].message
                tool_calls = getattr(message, "tool_calls", []) or []
                if not tool_calls:
                    # No more tool calls, assume final answer
                    if message.content:
                        loop_actions.append({"type": "final_response", "content": message.content})
                    break
                # Handle tool calls
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name

                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except Exception as e:
                        return f"Error parsing tool arguments: {str(e)}"
                    server_path = self.tool_server_map.get(tool_name)
                    if server_path and server_path in self.sessions:
                        ctx.logger.info(f"Calling tool {tool_name} from {server_path}")
                        try:
                            result = await asyncio.wait_for(
                                self.sessions[server_path].call_tool(tool_name, tool_args),
                                timeout=self.default_timeout.total_seconds()
                            )
                            if isinstance(result.content, str):
                                tool_response = result.content
                            elif isinstance(result.content, list):
                                tool_response = "\n".join([str(item) for item in result.content])
                            else:
                                tool_response = str(result.content)
                            # Add tool call and result to messages
                            messages.append({
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [tool_call]
                            })
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id if hasattr(tool_call, 'id') else None,
                                "name": tool_name,
                                "content": tool_response
                            })
                            loop_actions.append({
                                "type": "tool_call",
                                "tool": tool_name,
                                "args": tool_args,
                                "result": tool_response
                            })
                        except asyncio.TimeoutError:
                            return f"Error: The MCP server did not respond. Please try again later."
                        except Exception as e:
                            return f"Error calling tool {tool_name}: {str(e)}"
                    else:
                        return f"Tool server for {tool_name} not found or not connected."
                
                # Summarize the actions and results
                summary_prompt = (
                    "Either continue calling tools as needed, or provide a final summary. "
                    "Make sure the all requested information is available if possible. It is "
                    "Recommended to check data using search before summarizing."
                )
                messages.append({"role": "user", "content": summary_prompt})
            print(messages)
            if len(loop_actions) and "type" in loop_actions[-1] and loop_actions[-1]["type"] == "final_response":
                return loop_actions[-1]["content"]
            else:
                return "Agent loop completed, but no summary was generated."
        except Exception as e:
            ctx.logger.error(f"Error processing query: {str(e)}")
            ctx.logger.error(f"Error type: {type(e)}")
            import traceback
            ctx.logger.error(f"Traceback: {traceback.format_exc()}")
            return f"An error occurred while processing your query: {str(e)}"

    async def cleanup(self):
        await self.exit_stack.aclose()

# Initialize chat protocol and agent
chat_proto = Protocol(spec=chat_protocol_spec)
mcp_agent = Agent(name='test-analysis-research-MCPagent', seed="arsh-test-analysis-research-agent-mcp", port=8001, mailbox=True, readme_path="README.md", publish_agent_details=True)
client = AnalysisResearchMCPClient()

@chat_proto.on_message(model=ChatMessage)
async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    try:
        ack = ChatAcknowledgement(
            timestamp=datetime.now(timezone.utc),
            acknowledged_msg_id=msg.msg_id
        )
        await ctx.send(sender, ack)

        if not client.sessions:
            await client.connect_to_servers(ctx)

        for item in msg.content:
            if isinstance(item, StartSessionContent):
                ctx.logger.info(f"Got a start session message from {sender}")
                continue
            elif isinstance(item, TextContent):
                ctx.logger.info(f"Got a message from {sender}: {item.text}")
                response_text = await client.process_query(item.text, ctx)
                ctx.logger.info(f"Response text: {response_text}")
                response = ChatMessage(
                    timestamp=datetime.now(timezone.utc),
                    msg_id=uuid4(),
                    content=[TextContent(type="text", text=response_text)]
                )
                await ctx.send(sender, response)
            else:
                ctx.logger.info(f"Got unexpected content from {sender}")
    except Exception as e:
        ctx.logger.error(f"Error handling chat message: {str(e)}")
        ctx.logger.error(f"Error type: {type(e)}")
        import traceback
        ctx.logger.error(f"Traceback: {traceback.format_exc()}")
        error_response = ChatMessage(
            timestamp=datetime.now(timezone.utc),
            msg_id=uuid4(),
            content=[TextContent(type="text", text=f"An error occurred: {str(e)}")]
        )
        await ctx.send(sender, error_response)

@chat_proto.on_message(model=ChatAcknowledgement)
async def handle_chat_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"Received acknowledgement from {sender} for message {msg.acknowledged_msg_id}")
    if msg.metadata:
        ctx.logger.info(f"Metadata: {msg.metadata}")

mcp_agent.include(chat_proto)

if __name__ == "__main__":
    try:
        fund_agent_if_low(mcp_agent.wallet.address())
        mcp_agent.run()
    except Exception as e:
        print(f"Error running agent: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
    finally:
        asyncio.run(client.cleanup())