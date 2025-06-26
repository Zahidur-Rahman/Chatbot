import logging
import os
import asyncio
import subprocess
import sys
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from contextlib import asynccontextmanager
import psycopg2
import json
import threading
import queue
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Set Windows event loop policy
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Validate required environment variables
def validate_env_vars():
    """Validate that required environment variables are set"""
    required_vars = {
        "POSTGRES_HOST": os.getenv("POSTGRES_HOST", "localhost"),
        "POSTGRES_DB": os.getenv("POSTGRES_DB", "resturent"),
        "POSTGRES_USER": os.getenv("POSTGRES_USER", "postgres"),
        "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD", "postgres"),
        "POSTGRES_PORT": os.getenv("POSTGRES_PORT", "5432"),
        "MISTRAL_API_KEY": os.getenv("MISTRAL_API_KEY")
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
    
    return required_vars

# Initialize environment validation
env_vars = validate_env_vars()

# Global variables for MCP
mcp_client = None
mcp_process = None
class SimpleMCPClient:
    """Simplified MCP client for Windows compatibility"""
    
    def __init__(self, process):
        self.process = process
        self.tools = [
            {"name": "execute_query", "description": "Execute a SQL query and return results"},
            {"name": "get_table_schema", "description": "Get schema information for a specific table"},
            {"name": "list_tables", "description": "List all tables in the database"}
        ]
        self._message_id = 0
        self._request_queue = queue.Queue()
        self._response_queue = queue.Queue()
        self._communication_thread = None
        self._start_communication_thread()
        self._initialized = False
        self._initialize_server()
    
    def _initialize_server(self):
        """Initialize the MCP server with proper handshake"""
        try:
            # Send initialization request
            init_message = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "clientInfo": {
                        "name": "fastapi-client",
                        "version": "1.0.0"
                    }
                }
            }
            
            init_str = json.dumps(init_message) + "\n"
            self.process.stdin.write(init_str)
            self.process.stdin.flush()
            
            # Wait for initialization response
            response_line = self.process.stdout.readline()
            if response_line:
                try:
                    response = json.loads(response_line.strip())
                    if response.get("id") == 1 and "result" in response:
                        logger.info("✅ MCP server initialized successfully")
                        self._initialized = True
                        
                        # Send initialized notification
                        initialized_notification = {
                            "jsonrpc": "2.0",
                            "method": "notifications/initialized"
                        }
                        notif_str = json.dumps(initialized_notification) + "\n"
                        self.process.stdin.write(notif_str)
                        self.process.stdin.flush()
                    else:
                        logger.error(f"Initialization failed: {response}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse initialization response: {e}")
            else:
                logger.error("No initialization response received")
                
        except Exception as e:
            logger.error(f"Initialization error: {e}")
    
    def _start_communication_thread(self):
        """Start a background thread to handle MCP communication"""
        def communication_worker():
            try:
                while True:
                    try:
                        # Check for requests
                        try:
                            request = self._request_queue.get(timeout=1)
                            if request is None:  # Poison pill to stop thread
                                break
                            
                            # Send request to MCP server
                            request_str = json.dumps(request) + "\n"
                            self.process.stdin.write(request_str)
                            self.process.stdin.flush()
                            
                            # Read response
                            response_line = self.process.stdout.readline()
                            if response_line:
                                try:
                                    response = json.loads(response_line.strip())
                                    self._response_queue.put(response)
                                except json.JSONDecodeError:
                                    self._response_queue.put({"error": "Invalid JSON response"})
                            else:
                                self._response_queue.put({"error": "No response from server"})
                                
                        except queue.Empty:
                            # Check if process is still alive
                            if self.process.poll() is not None:
                                logger.error("MCP process died")
                                break
                            continue
                            
                    except Exception as e:
                        logger.error(f"Communication thread error: {e}")
                        self._response_queue.put({"error": str(e)})
                        
            except Exception as e:
                logger.error(f"Communication worker fatal error: {e}")
        
        self._communication_thread = threading.Thread(target=communication_worker, daemon=True)
        self._communication_thread.start()
    
    async def get_tools(self):
        """Get available tools from the MCP server"""
        if not self._initialized:
            return self.tools
            
        try:
            self._message_id += 1
            message = {
                "jsonrpc": "2.0",
                "id": self._message_id,
                "method": "tools/list"
            }
            
            # Send request
            self._request_queue.put(message)
            
            # Wait for response with timeout
            timeout = 10
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    response = self._response_queue.get(timeout=1)
                    if response.get("id") == self._message_id:
                        if "result" in response:
                            return response["result"].get("tools", self.tools)
                        elif "error" in response:
                            logger.error(f"Tools list error: {response['error']}")
                            return self.tools
                except queue.Empty:
                    continue
            
            logger.warning("Tools list request timeout")
            return self.tools
            
        except Exception as e:
            logger.error(f"Error getting tools: {e}")
            return self.tools
    
    async def call_tool(self, tool_name: str, arguments: dict):
        """Call a tool on the MCP server"""
        if not self._initialized:
            return {"error": "MCP server not initialized"}
            
        try:
            self._message_id += 1
            message = {
                "jsonrpc": "2.0",
                "id": self._message_id,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            # Send request
            self._request_queue.put(message)
            
            # Wait for response with timeout
            timeout = 30  # 30 seconds timeout
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    response = self._response_queue.get(timeout=1)
                    if response.get("id") == self._message_id:
                        if "result" in response:
                            return response["result"]
                        elif "error" in response:
                            return {"error": response["error"]}
                except queue.Empty:
                    continue
            
            return {"error": "Request timeout"}
            
        except Exception as e:
            logger.error(f"Error calling MCP tool {tool_name}: {e}")
            return {"error": str(e)}
    
    async def aclose(self):
        """Close the MCP client"""
        if self._communication_thread:
            self._request_queue.put(None)  # Poison pill
            self._communication_thread.join(timeout=5)
        
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                self.process.kill()

# Test database connection
async def test_database_connection():
    """Test database connection and log status"""
    try:
        with psycopg2.connect(
            host=env_vars["POSTGRES_HOST"],
            database=env_vars["POSTGRES_DB"],
            user=env_vars["POSTGRES_USER"],
            password=env_vars["POSTGRES_PASSWORD"],
            port=env_vars["POSTGRES_PORT"]
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT version();")
                version = cursor.fetchone()[0]
                logger.info(f"✅ Database connection successful: {version}")
                return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

# Direct database functions as fallback
async def execute_direct_query(query: str):
    """Execute a SQL query directly against the database"""
    try:
        logger.info(f"🔍 Executing direct query: {query[:100]}...")
        
        # Basic input validation
        query = query.strip()
        if not query:
            return {"error": "Empty query"}
        
        # FIXED: Only prevent dangerous operations that are NOT SELECT
        query_upper = query.upper().strip()
        if not query_upper.startswith('SELECT'):
            dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE', 'GRANT', 'REVOKE']
            for keyword in dangerous_keywords:
                if query_upper.startswith(keyword):
                    return {"error": f"Operation not allowed: {keyword}"}
        
        with psycopg2.connect(
            host=env_vars["POSTGRES_HOST"],
            database=env_vars["POSTGRES_DB"],
            user=env_vars["POSTGRES_USER"],
            password=env_vars["POSTGRES_PASSWORD"],
            port=env_vars["POSTGRES_PORT"]
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                if query.strip().upper().startswith('SELECT'):
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    logger.info(f"✅ Query executed successfully, returned {len(rows)} rows")
                    return {"columns": columns, "rows": rows}
                else:
                    conn.commit()
                    logger.info(f"✅ Query executed successfully, {cursor.rowcount} rows affected")
                    return {"message": f"Query executed successfully. Rows affected: {cursor.rowcount}"}
    except Exception as e:
        logger.error(f"❌ Database query error: {e}")
        return {"error": str(e)}

async def get_table_schema_direct(table_name: str):
    """Get schema information for a table directly"""
    try:
        with psycopg2.connect(
            host=env_vars["POSTGRES_HOST"],
            database=env_vars["POSTGRES_DB"],
            user=env_vars["POSTGRES_USER"],
            password=env_vars["POSTGRES_PASSWORD"],
            port=env_vars["POSTGRES_PORT"]
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_name = %s AND table_schema = 'public'
                    ORDER BY ordinal_position
                """, (table_name,))
                columns = cursor.fetchall()
                return [{"name": col[0], "type": col[1], "nullable": col[2], "default": col[3]} for col in columns]
    except Exception as e:
        return {"error": str(e)}

async def start_mcp_server():
    """Start the MCP server process"""
    global mcp_process
    try:
        logger.info("🚀 Starting MCP server process...")
        
        # Use the postgres_server.py file
        script_name = "postgres_server.py"
        
        # Check if file exists
        if not os.path.exists(script_name):
            logger.warning(f"⚠️ MCP server file {script_name} not found, will use direct database access")
            return None
        
        # Start the MCP server as a separate process
        mcp_process = subprocess.Popen(
            [sys.executable, script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            text=True,
            bufsize=0,  # Unbuffered for Windows
            universal_newlines=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform.startswith('win') else 0
        )
        
        logger.info(f"✅ MCP server process started with PID: {mcp_process.pid}")
        
        # Give it more time to start up on Windows
        await asyncio.sleep(3)
        
        # Check if process is still running
        if mcp_process.poll() is not None:
            # Process died, get error output
            logger.info("Hello Problem")
            stdout, stderr = mcp_process.communicate()
            logger.error(f"❌ MCP server process died. stdout: {stdout}, stderr: {stderr}")
            raise Exception(f"MCP server process died. stdout: {stdout}, stderr: {stderr}")
        
        logger.info("✅ MCP server is running successfully")
        return mcp_process
        
    except Exception as e:
        logger.error(f"❌ Failed to start MCP server process: {e}")
        if mcp_process:
            try:
                mcp_process.terminate()
            except:
                pass
            mcp_process = None
        return None

async def get_mcp_client():
    """Get or create MCP client"""
    global mcp_client, mcp_process
    
    if mcp_client is None:
        # Try to start MCP server if not running
        if mcp_process is None or mcp_process.poll() is not None:
            mcp_process = await start_mcp_server()
        
        if mcp_process:
            try:
                mcp_client = SimpleMCPClient(mcp_process)
                logger.info("✅ MCP client created successfully")
                
                # Test the client
                await asyncio.sleep(1)
                tools = await mcp_client.get_tools()
                logger.info(f"✅ MCP client working, tools available: {[tool['name'] for tool in tools]}")
                
            except Exception as e:
                logger.error(f"❌ Failed to create MCP client: {e}")
                mcp_client = None
    
    return mcp_client

def is_database_related_query(message: str) -> bool:
    """Check if the message is database/SQL related"""
    database_keywords = [
        'select', 'query', 'database', 'table', 'sql', 'data', 'show', 'find', 'get', 'fetch',
        'customer', 'order', 'product', 'restaurant', 'menu', 'count', 'list', 'search',
        'where', 'from', 'join', 'group', 'having', 'order by', 'limit'
    ]
    
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in database_keywords)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    try:
        logger.info("🚀 Starting application...")
        
        # Test database connection on startup
        db_connected = await test_database_connection()
        if not db_connected:
            logger.error("❌ Database connection failed on startup")
        
        # Try to initialize MCP
        client = await get_mcp_client()
        if client:
            logger.info("✅ MCP tools initialized successfully")
        else:
            logger.warning("⚠️ MCP tools not available, using direct database access")
        
        logger.info("✅ Application startup completed")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
    
    yield
    
    logger.info("🛑 Application shutting down")
    # Clean up MCP client and process
    global mcp_client, mcp_process
    if mcp_client:
        try:
            await mcp_client.aclose()
        except Exception as e:
            logger.error(f"Error closing MCP client: {e}")
    if mcp_process:
        try:
            mcp_process.terminate()
            mcp_process.wait(timeout=5)
        except Exception as e:
            logger.error(f"Error stopping MCP process: {e}")
            try:
                mcp_process.kill()
            except:
                pass

app = FastAPI(title="MCP Chatbot API", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    response: str
    tools_used: List[str] = []

class QueryRequest(BaseModel):
    query: str

# Initialize Mistral model
def get_mistral_model():
    api_key = env_vars["MISTRAL_API_KEY"]
    if not api_key:
        raise HTTPException(status_code=500, detail="MISTRAL_API_KEY not found")
    return ChatMistralAI(model="mistral-large-2407", api_key=api_key)

# Helper to fetch and cache schema info for all tables
async def get_all_table_schemas():
    """Fetch schema info for all tables in the public schema and return as a dict."""
    try:
        with psycopg2.connect(
            host=env_vars["POSTGRES_HOST"],
            database=env_vars["POSTGRES_DB"],
            user=env_vars["POSTGRES_USER"],
            password=env_vars["POSTGRES_PASSWORD"],
            port=env_vars["POSTGRES_PORT"]
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """)
                tables = [row[0] for row in cursor.fetchall()]
                schema_info = {}
                for table in tables:
                    cursor.execute("""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = %s AND table_schema = 'public' 
                        ORDER BY ordinal_position
                    """, (table,))
                    columns = [f"{row[0]} ({row[1]})" for row in cursor.fetchall()]
                    schema_info[table] = columns
                return schema_info
    except Exception as e:
        logger.error(f"Error getting table schemas: {e}")
        return {}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    try:
        logger.info(f"💬 Processing chat request: {request.message}")
        
        # Check if this is a database-related query
        if not is_database_related_query(request.message):
            logger.info("🗣️ Non-database query detected, using general chat")
            return await general_chat_processing(request)
        
        logger.info("🗄️ Database query detected, processing with SQL")
        
        # Try MCP first, then fallback to direct access
        client = await get_mcp_client()
        if client:
            logger.info("🔧 Using MCP tools for processing")
            return await mcp_chat_processing(request, client)
        else:
            logger.info("🔄 MCP not available, using direct database access")
            return await fallback_chat_processing(request)
        
    except Exception as e:
        logger.error(f"❌ Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

async def general_chat_processing(request: ChatRequest):
    """Handle general non-database chat"""
    try:
        model = get_mistral_model()
        
        # Build conversation context
        messages = []
        for msg in request.conversation_history[-5:]:  # Last 5 messages for context
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            else:
                messages.append(SystemMessage(content=msg.content))
        
        # Add current message
        messages.append(HumanMessage(content=request.message))
        
        response = await model.ainvoke(messages)
        
        return ChatResponse(
            response=response.content,
            tools_used=["general_chat"]
        )
        
    except Exception as e:
        logger.error(f"General chat error: {e}")
        return ChatResponse(
            response="I'm having trouble processing your request right now. Please try again.",
            tools_used=["error"]
        )

async def mcp_chat_processing(request: ChatRequest, client):
    """Chat processing using MCP tools"""
    try:
        # Get table schemas using MCP
        logger.info("📊 Getting table schemas via MCP")
        tables_result = await client.call_tool("list_tables", {})
        
        # Handle different response formats
        if isinstance(tables_result, dict):
            if 'error' in tables_result:
                logger.error(f"Error getting tables: {tables_result['error']}")
                return await fallback_chat_processing(request)
            
            # Check if it's a direct result or wrapped in content
            if 'content' in tables_result:
                # Extract content from MCP response
                content = tables_result['content']
                if isinstance(content, list) and len(content) > 0:
                    tables_data = json.loads(content[0]['text']) if content[0].get('text') else {}
                else:
                    tables_data = {}
            else:
                # Direct result
                tables_data = tables_result
        else:
            logger.error(f"Unexpected tables result format: {type(tables_result)}")
            return await fallback_chat_processing(request)
        
        # Build schema context
        schema_text = "Available tables:\n"
        if isinstance(tables_data, dict) and 'tables' in tables_data:
            for table in tables_data['tables']:
                schema_result = await client.call_tool("get_table_schema", {"table_name": table['name']})
                
                if isinstance(schema_result, dict):
                    if 'error' in schema_result:
                        continue
                    
                    # Handle schema response format
                    if 'content' in schema_result:
                        content = schema_result['content']
                        if isinstance(content, list) and len(content) > 0:
                            schema_data = json.loads(content[0]['text']) if content[0].get('text') else {}
                        else:
                            schema_data = {}
                    else:
                        schema_data = schema_result
                    
                    if isinstance(schema_data, dict) and 'columns' in schema_data:
                        columns = [f"{col['name']} ({col['type']})" for col in schema_data['columns']]
                        schema_text += f"{table['name']}: {', '.join(columns)}\n"
        
        # Generate SQL using AI
        model = get_mistral_model()
        improved_prompt = (
            "You are an expert SQL assistant for a PostgreSQL database. "
            "Your job is to convert user requests into a single, safe, syntactically correct SQL SELECT query.\n"
            "- Only generate SELECT statements. Never use DROP, DELETE, TRUNCATE, ALTER, CREATE, INSERT, or UPDATE.\n"
            f"- Use the following table schemas:\n{schema_text}\n"
            "- Do NOT use Markdown formatting or code blocks. Only output the SQL statement, nothing else.\n"
            "- If the user asks for something not possible with a SELECT, reply: 'Operation not allowed.'\n"
            "- Use only the columns and tables provided.\n"
            f"User request: {request.message}"
        )
        
        sql_response = await model.ainvoke([HumanMessage(content=improved_prompt)])
        sql_query = sql_response.content.strip().split('\n')[0]
        sql_query = sql_query.replace('\\_', '_').replace('\\', '')
        
        if sql_query == "Operation not allowed.":
            return ChatResponse(response="Operation not allowed.", tools_used=["mcp_tools"])
        
        # Execute query using MCP
        logger.info(f"🔍 Executing SQL via MCP: {sql_query}")
        query_result = await client.call_tool("execute_query", {"query": sql_query})
        
        # Handle query result
        if isinstance(query_result, dict):
            if 'error' in query_result:
                return ChatResponse(
                    response=f"Generated SQL: {sql_query}\n\nError: {query_result['error']}",
                    tools_used=["mcp_tools"]
                )
            
            # Handle different response formats
            if 'content' in query_result:
                content = query_result['content']
                if isinstance(content, list) and len(content) > 0:
                    result_data = json.loads(content[0]['text']) if content[0].get('text') else {}
                else:
                    result_data = {}
            else:
                result_data = query_result
            
            # Format response
            if isinstance(result_data, dict):
                if 'error' in result_data:
                    return ChatResponse(
                        response=f"Generated SQL: {sql_query}\n\nError: {result_data['error']}",
                        tools_used=["mcp_tools"]
                    )
                elif 'results' in result_data:
                    formatted_result = f"Query Results:\nRows returned: {result_data.get('row_count', 0)}\n"
                    if result_data['results']:
                        formatted_result += "Data:\n"
                        for i, row in enumerate(result_data['results'][:10]):
                            formatted_result += f"  {i+1}. {row}\n"
                        if len(result_data['results']) > 10:
                            formatted_result += f"  ... and {len(result_data['results']) - 10} more rows\n"
                    
                    return ChatResponse(
                        response=f"SQL Query: {sql_query}\n\n{formatted_result}",
                        tools_used=["mcp_tools"]
                    )
        
        return ChatResponse(
            response=f"SQL Query: {sql_query}\n\nResult: {str(query_result)}",
            tools_used=["mcp_tools"]
        )
        
    except Exception as e:
        logger.error(f"❌ MCP chat processing error: {e}")
        # Fallback to direct access
        return await fallback_chat_processing(request)
    """Chat processing using MCP tools"""
    try:
        # Get table schemas using MCP
        logger.info("📊 Getting table schemas via MCP")
        tables_result = await client.call_tool("list_tables", {})
        
        if isinstance(tables_result, list) and len(tables_result) > 0:
            tables_data = json.loads(tables_result[0].text) if hasattr(tables_result[0], 'text') else tables_result[0]
        else:
            tables_data = tables_result
        
        # Build schema context
        schema_text = "Available tables:\n"
        if isinstance(tables_data, dict) and 'tables' in tables_data:
            for table in tables_data['tables']:
                schema_result = await client.call_tool("get_table_schema", {"table_name": table['name']})
                if isinstance(schema_result, list) and len(schema_result) > 0:
                    schema_data = json.loads(schema_result[0].text) if hasattr(schema_result[0], 'text') else schema_result[0]
                    if isinstance(schema_data, dict) and 'columns' in schema_data:
                        columns = [f"{col['name']} ({col['type']})" for col in schema_data['columns']]
                        schema_text += f"{table['name']}: {', '.join(columns)}\n"
        
        # Generate SQL using AI
        model = get_mistral_model()
        improved_prompt = (
            "You are an expert SQL assistant for a PostgreSQL database. "
            "Your job is to convert user requests into a single, safe, syntactically correct SQL SELECT query.\n"
            "- Only generate SELECT statements. Never use DROP, DELETE, TRUNCATE, ALTER, CREATE, INSERT, or UPDATE.\n"
            f"- Use the following table schemas:\n{schema_text}\n"
            "- Do NOT use Markdown formatting or code blocks. Only output the SQL statement, nothing else.\n"
            "- If the user asks for something not possible with a SELECT, reply: 'Operation not allowed.'\n"
            "- Use only the columns and tables provided.\n"
            f"User request: {request.message}"
        )
        
        sql_response = await model.ainvoke([HumanMessage(content=improved_prompt)])
        sql_query = sql_response.content.strip().split('\n')[0]
        sql_query = sql_query.replace('\\_', '_').replace('\\', '')
        
        if sql_query == "Operation not allowed.":
            return ChatResponse(response="Operation not allowed.", tools_used=["mcp_tools"])
        
        # Execute query using MCP
        logger.info(f"🔍 Executing SQL via MCP: {sql_query}")
        query_result = await client.call_tool("execute_query", {"query": sql_query})
        
        if isinstance(query_result, list) and len(query_result) > 0:
            result_data = json.loads(query_result[0].text) if hasattr(query_result[0], 'text') else query_result[0]
        else:
            result_data = query_result
        
        # Format response
        if isinstance(result_data, dict):
            if 'error' in result_data:
                return ChatResponse(
                    response=f"Generated SQL: {sql_query}\n\nError: {result_data['error']}",
                    tools_used=["mcp_tools"]
                )
            elif 'results' in result_data:
                formatted_result = f"Query Results:\nRows returned: {result_data.get('row_count', 0)}\n"
                if result_data['results']:
                    formatted_result += "Data:\n"
                    for i, row in enumerate(result_data['results'][:10]):
                        formatted_result += f"  {i+1}. {row}\n"
                    if len(result_data['results']) > 10:
                        formatted_result += f"  ... and {len(result_data['results']) - 10} more rows\n"
                
                return ChatResponse(
                    response=f"SQL Query: {sql_query}\n\n{formatted_result}",
                    tools_used=["mcp_tools"]
                )
        
        return ChatResponse(
            response=f"SQL Query: {sql_query}\n\nResult: {str(result_data)}",
            tools_used=["mcp_tools"]
        )
        
    except Exception as e:
        logger.error(f"❌ MCP chat processing error: {e}")
        # Fallback to direct access
        return await fallback_chat_processing(request)

async def fallback_chat_processing(request: ChatRequest):
    """Chat processing using direct database access"""
    try:
        schema_info = await get_all_table_schemas()
        schema_text = "\n".join([
            f"{table}: {', '.join(columns)}" for table, columns in schema_info.items()
        ])
        
        model = get_mistral_model()
        improved_prompt = (
            "You are an expert SQL assistant for a PostgreSQL database. "
            "Your job is to convert user requests into a single, safe, syntactically correct SQL SELECT query.\n"
            "- Only generate SELECT statements. Never use DROP, DELETE, TRUNCATE, ALTER, CREATE, INSERT, or UPDATE.\n"
            f"- Use the following table schemas:\n{schema_text}\n"
            "- Do NOT use Markdown formatting or code blocks. Only output the SQL statement, nothing else.\n"
            "- If the user asks for something not possible with a SELECT, reply: 'Operation not allowed.'\n"
            "- Use only the columns and tables provided.\n"
            f"User request: {request.message}"
        )
        
        sql_response = await model.ainvoke([HumanMessage(content=improved_prompt)])
        sql_query = sql_response.content.strip().split('\n')[0]
        sql_query = sql_query.replace('\\_', '_').replace('\\', '')
        
        if sql_query == "Operation not allowed.":
            return ChatResponse(response="Operation not allowed.", tools_used=["direct_query"])
        
        db_result = await execute_direct_query(sql_query)
        
        if isinstance(db_result, dict) and 'error' in db_result:
            return ChatResponse(
                response=f"Generated SQL: {sql_query}\n\nError executing query: {db_result['error']}",
                tools_used=["direct_query"]
            )
        
        # Format the result nicely
        if isinstance(db_result, dict) and 'columns' in db_result and 'rows' in db_result:
            formatted_result = "Query Results:\n"
            formatted_result += f"Columns: {', '.join(db_result['columns'])}\n"
            formatted_result += f"Rows: {len(db_result['rows'])}\n"
            if db_result['rows']:
                formatted_result += "Data:\n"
                for i, row in enumerate(db_result['rows'][:10]):  # Show first 10 rows
                    formatted_result += f"  {i+1}. {row}\n"
                if len(db_result['rows']) > 10:
                    formatted_result += f"  ... and {len(db_result['rows']) - 10} more rows\n"
        else:
            formatted_result = str(db_result)
        
        return ChatResponse(
            response=f"SQL Query: {sql_query}\n\n{formatted_result}",
            tools_used=["direct_query"]
        )
        
    except Exception as e:
        logger.error(f"❌ Chat processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")

@app.get("/tables")
async def get_tables():
    """Get list of tables in database"""
    try:
        logger.info("📋 Getting tables from database")
        
        with psycopg2.connect(
            host=env_vars["POSTGRES_HOST"],
            database=env_vars["POSTGRES_DB"],
            user=env_vars["POSTGRES_USER"],
            password=env_vars["POSTGRES_PASSWORD"],
            port=env_vars["POSTGRES_PORT"]
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT table_name, table_type
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """)
                tables = [{"name": row[0], "type": row[1]} for row in cursor.fetchall()]
                
        logger.info(f"✅ Found {len(tables)} tables")
        return {"tables": tables, "count": len(tables)}
        
    except Exception as e:
        logger.error(f"❌ Error getting tables: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting tables: {str(e)}")

@app.get("/tables/{table_name}/schema")
async def get_table_schema(table_name: str):
    """Get schema for a specific table"""
    try:
        logger.info(f"🔍 Getting schema for table: {table_name}")
        
        schema = await get_table_schema_direct(table_name)
        
        if isinstance(schema, dict) and 'error' in schema:
            raise HTTPException(status_code=404, detail=schema['error'])
        
        return {"table_name": table_name, "columns": schema}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting schema for table {table_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting schema: {str(e)}")

@app.post("/query")
async def execute_query(request: QueryRequest):
    """Execute a raw SQL query"""
    try:
        logger.info(f"🔍 Executing raw query: {request.query}")
        
        result = await execute_direct_query(request.query)
        
        if isinstance(result, dict) and 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Query execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Query execution error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Windows-specific configuration
    if sys.platform.startswith('win'):
        # Use ProactorEventLoop on Windows for better subprocess handling
        logger.info("The Start Point")
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        logger.info("The End Point")
    
    logger.info("🚀 Starting FastAPI server...")
    uvicorn.run(
        "chatbot_api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=["./"],
        log_level="info"
    )