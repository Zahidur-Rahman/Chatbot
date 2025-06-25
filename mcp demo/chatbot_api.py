import logging
import os
import asyncio
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from contextlib import asynccontextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
import re
import time
from fastapi.responses import FileResponse
from collections import deque
import uuid

# FastMCP imports
from fastmcp import FastMCP

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Conversation Memory Manager
class ConversationMemory:
    def __init__(self, max_conversations: int = 30):
        self.max_conversations = max_conversations
        self.conversations = {}  # session_id -> deque of messages
    
    def add_message(self, session_id: str, role: str, content: str, sql_query: str = None):
        """Add a message to conversation history"""
        if session_id not in self.conversations:
            self.conversations[session_id] = deque(maxlen=self.max_conversations)
        
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "sql_query": sql_query
        }
        
        self.conversations[session_id].append(message)
    
    def get_conversation_context(self, session_id: str, last_n: int = 10) -> List[Dict]:
        """Get last N messages for context"""
        if session_id not in self.conversations:
            return []
        
        messages = list(self.conversations[session_id])
        return messages[-last_n:] if len(messages) > last_n else messages
    
    def get_formatted_context(self, session_id: str, last_n: int = 10) -> str:
        """Get formatted conversation context for LLM"""
        context = self.get_conversation_context(session_id, last_n)
        if not context:
            return ""
        
        formatted = "\n--- Previous Conversation Context ---\n"
        for msg in context:
            formatted += f"{msg['role'].upper()}: {msg['content']}\n"
            if msg.get('sql_query'):
                formatted += f"SQL: {msg['sql_query']}\n"
        formatted += "--- End Context ---\n"
        
        return formatted

# Database configuration
class DatabaseConfig:
    def __init__(self):
        self.host = os.getenv("POSTGRES_HOST", "localhost")
        self.database = os.getenv("POSTGRES_DB", "resturent")
        self.user = os.getenv("POSTGRES_USER", "postgres")
        self.password = os.getenv("POSTGRES_PASSWORD", "postgres")
        self.port = int(os.getenv("POSTGRES_PORT", "5432"))
        
        if not self.password or self.password == "postgres":
            logger.warning("Using default password. Consider setting POSTGRES_PASSWORD")
    
    def get_connection_params(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "database": self.database,
            "user": self.user,
            "password": self.password,
            "port": self.port
        }

db_config = DatabaseConfig()

# FastMCP Client Manager
class FastMCPManager:
    def __init__(self):
        self.mcp = None
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize FastMCP client"""
        try:
            logger.info("Initializing FastMCP client...")
            
            # Create FastMCP instance with database tools
            self.mcp = FastMCP("PostgreSQL Database Tools")
            
            # Register database tools
            await self._register_tools()
            
            # Directly test the tool registration
            result = await self.mcp.run_tool("list_tables", {})
            if "error" not in result:
                self.is_initialized = True
                logger.info("FastMCP client initialized successfully")
                return True
            else:
                logger.error(f"FastMCP test failed: {result}")
                return False
                
        except Exception as e:
            logger.error(f"FastMCP initialization error: {e}")
            return False
    
    async def _register_tools(self):
        """Register database tools with FastMCP"""
        
        @self.mcp.tool()
        async def list_tables() -> Dict[str, Any]:
            """List all tables in the database"""
            try:
                with psycopg2.connect(**db_config.get_connection_params()) as conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                        cursor.execute("""
                            SELECT 
                                t.table_name,
                                COALESCE(s.n_tup_ins - s.n_tup_del, 0) as estimated_rows,
                                obj_description(c.oid) as table_comment
                            FROM information_schema.tables t
                            LEFT JOIN pg_stat_user_tables s ON s.relname = t.table_name
                            LEFT JOIN pg_class c ON c.relname = t.table_name
                            WHERE t.table_schema = 'public' AND t.table_type = 'BASE TABLE'
                            ORDER BY t.table_name;
                        """)
                        tables = cursor.fetchall()
                        
                        return {
                            "tables": [
                                {
                                    "name": table['table_name'],
                                    "estimated_rows": table['estimated_rows'] if table['estimated_rows'] is not None else 0,
                                    "comment": table['table_comment']
                                }
                                for table in tables
                            ],
                            "count": len(tables)
                        }
            except Exception as e:
                return {"error": f"Database error: {str(e)}"}
        
        @self.mcp.tool()
        async def execute_query(query: str) -> Dict[str, Any]:
            """Execute a SQL SELECT query and return results"""
            try:
                # Validate query
                is_valid, message = self._validate_query(query)
                if not is_valid:
                    return {"error": message}
                
                with psycopg2.connect(**db_config.get_connection_params()) as conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                        cursor.execute(query)
                        results = cursor.fetchall()
                        
                        data = [dict(row) for row in results]
                        
                        return {
                            "columns": list(data[0].keys()) if data else [],
                            "rows": data,
                            "row_count": len(data),
                            "query": query
                        }
            except Exception as e:
                return {"error": f"Database error: {str(e)}"}
        
        @self.mcp.tool()
        async def get_table_schema(table_name: str) -> Dict[str, Any]:
            """Get schema information for a specific table"""
            try:
                with psycopg2.connect(**db_config.get_connection_params()) as conn:
                    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                        # Get column information
                        cursor.execute("""
                            SELECT 
                                column_name, 
                                data_type, 
                                is_nullable, 
                                column_default,
                                character_maximum_length,
                                numeric_precision,
                                numeric_scale,
                                ordinal_position
                            FROM information_schema.columns
                            WHERE table_name = %s AND table_schema = 'public'
                            ORDER BY ordinal_position;
                        """, (table_name,))
                        columns = cursor.fetchall()
                        
                        if not columns:
                            return {"error": f"Table '{table_name}' not found"}
                        
                        # Get primary keys
                        cursor.execute("""
                            SELECT kcu.column_name
                            FROM information_schema.table_constraints tc
                            JOIN information_schema.key_column_usage kcu 
                                ON tc.constraint_name = kcu.constraint_name
                            WHERE tc.table_name = %s 
                                AND tc.constraint_type = 'PRIMARY KEY'
                                AND tc.table_schema = 'public';
                        """, (table_name,))
                        primary_keys = [row['column_name'] for row in cursor.fetchall()]
                        
                        return {
                            "table_name": table_name,
                            "columns": [dict(col) for col in columns],
                            "primary_keys": primary_keys
                        }
            except Exception as e:
                return {"error": f"Database error: {str(e)}"}
        
        @self.mcp.tool()
        async def get_full_schema() -> Dict[str, Any]:
            """Get full schema: tables, columns, and foreign keys"""
            try:
                tables_result = await list_tables()
                if "error" in tables_result:
                    return tables_result
                
                full_schema = {"tables": {}}
                
                for table_info in tables_result["tables"]:
                    table_name = table_info["name"]
                    schema_result = await get_table_schema(table_name)
                    
                    if "error" not in schema_result:
                        full_schema["tables"][table_name] = schema_result["columns"]
                
                return full_schema
            except Exception as e:
                return {"error": f"Error getting full schema: {str(e)}"}
    
    def _validate_query(self, query: str) -> tuple[bool, str]:
        """Validate SQL query for safety"""
        query = query.strip()
        
        if not query:
            return False, "Empty query"
        
        dangerous_patterns = [
            r'\bDROP\b', r'\bDELETE\b', r'\bTRUNCATE\b', 
            r'\bALTER\b', r'\bCREATE\b', r'\bINSERT\b', 
            r'\bUPDATE\b', r'\bGRANT\b', r'\bREVOKE\b'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, f"Operation not allowed: {pattern.replace('\\b', '')}"
        
        if not query.upper().strip().startswith('SELECT'):
            return False, "Only SELECT statements are allowed"
        
        return True, "Valid"
    
    async def list_tables(self):
        """Call list_tables tool"""
        if not self.is_initialized:
            return {"error": "FastMCP not initialized"}
        
        try:
            # Call the registered tool
            result = await self.mcp.run_tool("list_tables", {})
            return result
        except Exception as e:
            return {"error": f"FastMCP tool error: {str(e)}"}
    
    async def execute_query(self, query: str):
        """Call execute_query tool"""
        if not self.is_initialized:
            return {"error": "FastMCP not initialized"}
        
        try:
            result = await self.mcp.run_tool("execute_query", {"query": query})
            return result
        except Exception as e:
            return {"error": f"FastMCP tool error: {str(e)}"}
    
    async def get_table_schema(self, table_name: str):
        """Call get_table_schema tool"""
        if not self.is_initialized:
            return {"error": "FastMCP not initialized"}
        
        try:
            result = await self.mcp.run_tool("get_table_schema", {"table_name": table_name})
            return result
        except Exception as e:
            return {"error": f"FastMCP tool error: {str(e)}"}
    
    async def get_full_schema(self):
        """Call get_full_schema tool"""
        if not self.is_initialized:
            return {"error": "FastMCP not initialized"}
        
        try:
            result = await self.mcp.run_tool("get_full_schema", {})
            return result
        except Exception as e:
            return {"error": f"FastMCP tool error: {str(e)}"}

# Initialize managers
mcp_manager = FastMCPManager()
conversation_memory = ConversationMemory()

# Database operations
class DatabaseManager:
    def __init__(self, config: DatabaseConfig):
        self.config = config
    
    def get_connection(self):
        """Get database connection"""
        try:
            return psycopg2.connect(**self.config.get_connection_params())
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    async def test_connection(self) -> bool:
        """Test database connectivity"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def validate_query(self, query: str) -> tuple[bool, str]:
        """Validate SQL query for safety"""
        query = query.strip()
        
        if not query:
            return False, "Empty query"
        
        dangerous_patterns = [
            r'\bDROP\b', r'\bDELETE\b', r'\bTRUNCATE\b', 
            r'\bALTER\b', r'\bCREATE\b', r'\bINSERT\b', 
            r'\bUPDATE\b', r'\bGRANT\b', r'\bREVOKE\b'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, f"Operation not allowed: {pattern.replace('\\b', '')}"
        
        if not query.upper().strip().startswith('SELECT'):
            return False, "Only SELECT statements are allowed"
        
        return True, "Valid"
    
    async def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute SQL query with proper error handling"""
        try:
            is_valid, message = self.validate_query(query)
            if not is_valid:
                return {"error": message}
            
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query)
                    
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []
                    return {
                        "columns": columns,
                        "rows": [dict(row) for row in rows],
                        "row_count": len(rows)
                    }
        
        except psycopg2.Error as e:
            logger.error(f"Database query error: {e}")
            return {"error": f"Database error: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected query error: {e}")
            return {"error": f"Unexpected error: {str(e)}"}

db_manager = DatabaseManager(db_config)

# Pydantic models
class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=10000)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_history: Optional[List[ChatMessage]] = Field(default_factory=list)

class ChatResponse(BaseModel):
    response: str
    session_id: str
    sql_query: Optional[str] = None
    tools_used: List[str] = Field(default_factory=list)
    execution_time: Optional[float] = None
    conversation_context_used: bool = False

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)

# SQL Generator
class SQLGenerator:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.model = None
        self._schema_cache = {}
        self._cache_expiry = 300
        self._last_cache_update = 0
    
    def get_model(self):
        """Get or create Mistral model"""
        if not self.model:
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise HTTPException(status_code=500, detail="MISTRAL_API_KEY not configured")
            self.model = ChatMistralAI(
                model="mistral-large-2407",
                api_key=api_key,
                temperature=0.1
            )
        return self.model
    
    async def get_cached_schemas(self):
        """Get schemas with caching via FastMCP"""
        current_time = time.time()
        
        if (current_time - self._last_cache_update) > self._cache_expiry:
            # Try to get schema via FastMCP first
            result = await mcp_manager.get_full_schema()
            if "error" not in result and "tables" in result:
                schemas = {}
                tables_data = result.get("tables", {})
                for table_name, columns in tables_data.items():
                    schemas[table_name] = [
                        f"{col['column_name']} ({col['data_type']})"
                        for col in columns
                    ]
                self._schema_cache = schemas
            else:
                # Fallback to direct database access
                self._schema_cache = await self._get_schemas_direct()
            
            self._last_cache_update = current_time
            
        return self._schema_cache
    
    async def _get_schemas_direct(self):
        """Fallback method to get schemas directly from database"""
        try:
            with db_manager.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                    """)
                    tables = cursor.fetchall()
                    
                    schemas = {}
                    for table in tables:
                        table_name = table['table_name']
                        cursor.execute("""
                            SELECT column_name, data_type 
                            FROM information_schema.columns 
                            WHERE table_name = %s AND table_schema = 'public'
                            ORDER BY ordinal_position
                        """, (table_name,))
                        columns = cursor.fetchall()
                        schemas[table_name] = [
                            f"{col['column_name']} ({col['data_type']})"
                            for col in columns
                        ]
                    
                    return schemas
        except Exception as e:
            logger.error(f"Error getting schemas directly: {e}")
            return {}
    
    async def generate_sql(self, user_message: str, session_id: str, conversation_history: List[ChatMessage] = None) -> tuple[str, bool]:
        """Generate SQL query with conversation context"""
        try:
            # Get conversation context
            context = conversation_memory.get_formatted_context(session_id, last_n=10)
            context_used = bool(context)
            
            # Get schema information
            schemas = await self.get_cached_schemas()
            
            if not schemas:
                return "I couldn't retrieve the database structure. Please check the database connection.", False
            
            # Check if this is a database question
            db_keywords = ['select', 'table', 'database', 'query', 'data', 'show', 'find', 'get', 'list', 'count', 'sum', 'average', 'group', 'order', 'where']
            if not any(keyword in user_message.lower() for keyword in db_keywords):
                # General conversation with context
                model = self.get_model()
                messages = [
                    SystemMessage(content="You are a helpful AI assistant. Use the conversation context to provide relevant responses.")
                ]
                
                if context:
                    messages.append(SystemMessage(content=context))
                
                messages.append(HumanMessage(content=user_message))
                response = await model.ainvoke(messages)
                return response.content, context_used
            
            # Create schema description
            schema_text = "\n".join([
                f"Table {table}: {', '.join(columns)}"
                for table, columns in schemas.items()
            ])
            
            # Enhanced prompt with conversation context
            system_prompt = f"""You are an expert PostgreSQL query generator. Your task is to convert natural language requests into syntactically correct SELECT queries.

IMPORTANT RULES:
1. ONLY generate SELECT statements - never use DROP, DELETE, INSERT, UPDATE, ALTER, CREATE, TRUNCATE
2. Output ONLY the SQL query - no explanations, no markdown, no code blocks
3. Use proper PostgreSQL syntax
4. Use table and column names exactly as provided in the schema
5. Consider the conversation context when generating queries
6. If the question is not database-related, respond conversationally

DATABASE SCHEMA:
{schema_text}

{context if context else ""}

Generate a SELECT query for the following request."""

            messages = [SystemMessage(content=system_prompt)]
            
            # Add recent conversation history
            if conversation_history:
                for msg in conversation_history[-5:]:
                    if msg.role == "user":
                        messages.append(HumanMessage(content=msg.content))
                    elif msg.role == "assistant":
                        messages.append(AIMessage(content=msg.content))
            
            messages.append(HumanMessage(content=f"Request: {user_message}"))
            
            model = self.get_model()
            response = await model.ainvoke(messages)
            
            logger.info(f"Raw LLM response: {response.content[:200]}...")
            
            # Clean up the response
            sql_query = response.content.strip()
            sql_query = re.sub(r'```sql\s*', '', sql_query)
            sql_query = re.sub(r'```\s*', '', sql_query)
            sql_query = re.sub(r'^(SQL|Query):\s*', '', sql_query, flags=re.IGNORECASE)
            
            return sql_query, context_used
            
        except Exception as e:
            logger.error(f"SQL generation error: {e}")
            return f"I encountered an error while processing your request: {str(e)}", False

sql_generator = SQLGenerator(db_manager)

# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    try:
        logger.info("Starting application...")
        
        if await db_manager.test_connection():
            logger.info("Database connection successful")
        else:
            logger.warning("Database connection failed - some features may not work")
        
        # Initialize FastMCP
        mcp_initialized = await mcp_manager.initialize()
        if mcp_initialized:
            logger.info("FastMCP client initialized successfully")
        else:
            logger.warning("FastMCP client initialization failed - falling back to direct DB access")
        
        logger.info("Application startup completed")
        yield
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        yield
    
    finally:
        logger.info("Shutting down application...")
        logger.info("Application shutdown completed")

# FastAPI application
app = FastAPI(
    title="Chatbot with FastMCP",
    description="AI-powered chatbot with conversation memory and FastMCP for querying restaurant database",
    version="3.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# API endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Enhanced chat endpoint with conversation memory and FastMCP"""
    start_time = time.time()
    
    try:
        session_id = request.session_id or str(uuid.uuid4())
        logger.info(f"Processing chat request for session {session_id}: {request.message[:100]}...")
        
        # Add user message to conversation memory
        conversation_memory.add_message(session_id, "user", request.message)
        
        # Generate response with conversation context
        llm_response, context_used = await sql_generator.generate_sql(
            request.message, 
            session_id,
            request.conversation_history
        )
        
        # If the response looks like a SQL query, execute it
        if llm_response.strip().upper().startswith("SELECT"):
            # Try FastMCP first, then fallback to direct execution
            result = await mcp_manager.execute_query(llm_response)
            
            if "error" in result:
                # Fallback to direct database execution
                result = await db_manager.execute_query(llm_response)
            
            if "error" in result:
                response_text = f"Query error: {result['error']}"
                tools_used = ["sql_generator", "database_error"]
            else:
                if result.get("rows"):
                    response_text = f"Found {result.get('row_count', len(result.get('rows', [])))} results:\n\n"
                    response_text += json.dumps(result, indent=2, default=str)
                else:
                    response_text = "No results found for your query."
                
                tools_used = ["sql_generator", "fastmcp" if mcp_manager.is_initialized else "database"]
            
            # Add assistant response to memory
            conversation_memory.add_message(session_id, "assistant", response_text, llm_response)
            
            return ChatResponse(
                response=response_text,
                session_id=session_id,
                sql_query=llm_response,
                tools_used=tools_used,
                execution_time=time.time() - start_time,
                conversation_context_used=context_used
            )
        else:
            # Normal conversation response
            conversation_memory.add_message(session_id, "assistant", llm_response)
            
            return ChatResponse(
                response=llm_response,
                session_id=session_id,
                sql_query=None,
                tools_used=["llm"],
                execution_time=time.time() - start_time,
                conversation_context_used=context_used
            )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        error_response = f"An error occurred: {str(e)}"
        
        # Still add to memory for debugging
        if 'session_id' in locals():
            conversation_memory.add_message(session_id, "assistant", error_response)
        
        return ChatResponse(
            response=error_response,
            session_id=session_id if 'session_id' in locals() else str(uuid.uuid4()),
            tools_used=["error_handler"],
            execution_time=time.time() - start_time,
            conversation_context_used=False
        )

@app.get("/conversation/{session_id}")
async def get_conversation_history(session_id: str, last_n: int = 30):
    """Get conversation history for a session"""
    try:
        history = conversation_memory.get_conversation_context(session_id, last_n)
        return {
            "session_id": session_id,
            "messages": history,
            "count": len(history)
        }
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/conversation/{session_id}")
async def clear_conversation_history(session_id: str):
    """Clear conversation history for a session"""
    try:
        if session_id in conversation_memory.conversations:
            del conversation_memory.conversations[session_id]
        return {"message": f"Conversation history cleared for session {session_id}"}
    except Exception as e:
        logger.error(f"Error clearing conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Enhanced health check including FastMCP"""
    try:
        db_healthy = await db_manager.test_connection()
        
        mistral_healthy = True
        try:
            sql_generator.get_model()
        except Exception:
            mistral_healthy = False
        
        fastmcp_healthy = mcp_manager.is_initialized
        
        status = "healthy" if db_healthy and mistral_healthy else "degraded"
        
        return {
            "status": status,
            "timestamp": time.time(),
            "components": {
                "database": "healthy" if db_healthy else "unhealthy",
                "mistral_api": "healthy" if mistral_healthy else "unhealthy",
                "fastmcp": "healthy" if fastmcp_healthy else "unhealthy",
                "conversation_memory": "healthy"
            },
            "memory_stats": {
                "active_sessions": len(conversation_memory.conversations),
                "total_messages": sum(len(conv) for conv in conversation_memory.conversations.values())
            }
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "error",
            "timestamp": time.time(),
            "error": str(e)
        }

@app.post("/query")
async def execute_direct_query(request: QueryRequest):
    """Execute direct SQL query via FastMCP or database"""
    try:
        logger.info(f"Executing direct query: {request.query[:100]}...")
        
        # Try FastMCP first
        result = await mcp_manager.execute_query(request.query)
        
        if "error" in result:
            # Fallback to direct database execution
            result = await db_manager.execute_query(request.query)
        
        return {
            "success": "error" not in result,
            "result": result,
            "method": "fastmcp" if mcp_manager.is_initialized and "error" not in result else "direct"
        }
        
    except Exception as e:
        logger.error(f"Direct query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tables")
async def get_tables():
    """Get list of all tables via FastMCP"""
    try:
        result = await mcp_manager.list_tables()
        
        if "error" in result:
            # Fallback to direct database access
            with db_manager.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT 
                            table_name,
                            COALESCE(obj_description(c.oid), 'No description') as table_comment
                        FROM information_schema.tables t
                        LEFT JOIN pg_class c ON c.relname = t.table_name
                        WHERE t.table_schema = 'public' AND t.table_type = 'BASE TABLE'
                        ORDER BY t.table_name;
                    """)
                    tables = cursor.fetchall()
                    
                    result = {
                        "tables": [dict(table) for table in tables],
                        "count": len(tables)
                    }
        
        return result
        
    except Exception as e:
        logger.error(f"Get tables error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/schema/{table_name}")
async def get_table_schema_endpoint(table_name: str):
    """Get schema for specific table via FastMCP"""
    try:
        result = await mcp_manager.get_table_schema(table_name)
        
        if "error" in result:
            # Fallback to direct database access
            with db_manager.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT 
                            column_name, 
                            data_type, 
                            is_nullable, 
                            column_default,
                            character_maximum_length,
                            numeric_precision,
                            numeric_scale,
                            ordinal_position
                        FROM information_schema.columns
                        WHERE table_name = %s AND table_schema = 'public'
                        ORDER BY ordinal_position;
                    """, (table_name,))
                    columns = cursor.fetchall()
                    
                    if not columns:
                        raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")
                    
                    result = {
                        "table_name": table_name,
                        "columns": [dict(col) for col in columns]
                    }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get table schema error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/schema")
async def get_full_schema_endpoint():
    """Get full database schema via FastMCP"""
    try:
        result = await mcp_manager.get_full_schema()
        
        if "error" in result:
            # Fallback to direct database access
            schemas = await sql_generator._get_schemas_direct()
            result = {"tables": schemas}
        
        return result
        
    except Exception as e:
        logger.error(f"Get full schema error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        return {
            "conversation_memory": {
                "active_sessions": len(conversation_memory.conversations),
                "total_messages": sum(len(conv) for conv in conversation_memory.conversations.values()),
                "max_conversations": conversation_memory.max_conversations
            },
            "fastmcp": {
                "initialized": mcp_manager.is_initialized,
                "status": "healthy" if mcp_manager.is_initialized else "not_initialized"
            },
            "database": {
                "connected": await db_manager.test_connection(),
                "host": db_config.host,
                "database": db_config.database,
                "port": db_config.port
            },
            "cache": {
                "schema_cache_size": len(sql_generator._schema_cache),
                "last_cache_update": sql_generator._last_cache_update,
                "cache_expiry": sql_generator._cache_expiry
            }
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Chatbot API with FastMCP",
        "version": "3.0.0",
        "description": "AI-powered chatbot with conversation memory and FastMCP for querying restaurant database",
        "endpoints": {
            "chat": "POST /chat - Main chat interface",
            "health": "GET /health - Health check",
            "tables": "GET /tables - List all tables",
            "schema": "GET /schema - Get full schema",
            "table_schema": "GET /schema/{table_name} - Get specific table schema",
            "direct_query": "POST /query - Execute direct SQL query",
            "conversation_history": "GET /conversation/{session_id} - Get conversation history",
            "clear_conversation": "DELETE /conversation/{session_id} - Clear conversation history",
            "stats": "GET /stats - System statistics"
        },
        "features": [
            "Conversational AI with Mistral LLM",
            "FastMCP integration for database tools",
            "Conversation memory management",
            "SQL query generation and validation",
            "Database schema caching",
            "Health monitoring"
        ]
    }

# Additional utility endpoints for debugging
@app.get("/debug/mcp-status")
async def debug_mcp_status():
    """Debug endpoint for FastMCP status"""
    return {
        "initialized": mcp_manager.is_initialized,
        "mcp_instance": mcp_manager.mcp is not None,
        "available_tools": list(mcp_manager.mcp._tools.keys()) if mcp_manager.mcp and hasattr(mcp_manager.mcp, '_tools') else []
    }

@app.get("/debug/memory/{session_id}")
async def debug_session_memory(session_id: str):
    """Debug endpoint for session memory"""
    if session_id not in conversation_memory.conversations:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = list(conversation_memory.conversations[session_id])
    return {
        "session_id": session_id,
        "message_count": len(messages),
        "messages": messages,
        "formatted_context": conversation_memory.get_formatted_context(session_id)
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "detail": str(exc)}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return {"error": "Internal server error", "detail": "Please check the logs for more information"}

if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    
    logger.info(f"Starting server on {HOST}:{PORT}")
    
    uvicorn.run(
        "chatbot_api:app",
        host=HOST,
        port=PORT,
        reload=os.getenv("ENVIRONMENT", "production") == "development",
        log_level="info"
    )