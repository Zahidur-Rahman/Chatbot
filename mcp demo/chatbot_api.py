import logging
import os
import asyncio
import subprocess
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from contextlib import asynccontextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
import re
import json
from fastapi.responses import FileResponse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Database configuration with validation
class DatabaseConfig:
    def __init__(self):
        self.host = os.getenv("POSTGRES_HOST", "localhost")
        self.database = os.getenv("POSTGRES_DB", "resturent")  # Fixed typo
        self.user = os.getenv("POSTGRES_USER", "postgres")
        self.password = os.getenv("POSTGRES_PASSWORD", "postgres")
        self.port = int(os.getenv("POSTGRES_PORT", "5432"))
        
        # Validate required environment variables
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

# MCP client management
class MCPClientManager:
    def __init__(self):
        self.client = None
        self.process = None
        self.max_retries = 3
        self.retry_delay = 2
    
    async def initialize(self):
        """Initialize MCP client with retry logic"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Initializing MCP client (attempt {attempt + 1})")
                await self._start_mcp_server()
                logger.info("MCP client initialized successfully")
                return True
            except Exception as e:
                logger.warning(f"MCP initialization attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error("MCP client initialization failed after all attempts")
        return False
    
    async def _start_mcp_server(self):
        """Start MCP server process"""
        if self.process is None or self.process.poll() is not None:
            try:
                cmd = [sys.executable, "postgres_server.py"]
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )
                
                # Wait for server to start
                await asyncio.sleep(3)
                
                # Check if process is still running
                if self.process.poll() is not None:
                    stdout, stderr = self.process.communicate()
                    raise Exception(f"MCP server died. stdout: {stdout}, stderr: {stderr}")
                
                logger.info("MCP server process started successfully")
            except Exception as e:
                logger.error(f"Failed to start MCP server: {e}")
                self.process = None
                raise
    
    async def cleanup(self):
        """Clean up MCP resources"""
        if self.process:
            try:
                self.process.terminate()
                await asyncio.sleep(1)
                if self.process.poll() is None:
                    self.process.kill()
                logger.info("MCP server process terminated")
            except Exception as e:
                logger.error(f"Error cleaning up MCP process: {e}")

mcp_manager = MCPClientManager()

# Database operations with improved error handling
class DatabaseManager:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._connection_cache = None
    
    def get_connection(self):
        """Get database connection with connection pooling"""
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
        
        # Check for dangerous operations
        dangerous_patterns = [
            r'\bDROP\b', r'\bDELETE\b', r'\bTRUNCATE\b', 
            r'\bALTER\b', r'\bCREATE\b', r'\bINSERT\b', 
            r'\bUPDATE\b', r'\bGRANT\b', r'\bREVOKE\b'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False, f"Operation not allowed: {pattern.replace('\\b', '')}"
        
        # Ensure it's a SELECT statement
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
                    
                    if query.strip().upper().startswith('SELECT'):
                        rows = cursor.fetchall()
                        columns = [desc[0] for desc in cursor.description] if cursor.description else []
                        return {
                            "columns": columns,
                            "rows": [dict(row) for row in rows],
                            "row_count": len(rows)
                        }
                    else:
                        conn.commit()
                        return {
                            "message": f"Query executed successfully. Rows affected: {cursor.rowcount}",
                            "rows_affected": cursor.rowcount
                        }
        
        except psycopg2.Error as e:
            logger.error(f"Database query error: {e}")
            return {"error": f"Database error: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected query error: {e}")
            return {"error": f"Unexpected error: {str(e)}"}
    
    async def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for a specific table"""
        try:
            # Validate table name to prevent SQL injection
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
                return {"error": "Invalid table name"}
            
            query = """
                SELECT column_name, data_type, is_nullable, column_default,
                       character_maximum_length, numeric_precision, numeric_scale
                FROM information_schema.columns
                WHERE table_name = %s AND table_schema = 'public'
                ORDER BY ordinal_position
            """
            
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query, (table_name,))
                    columns = cursor.fetchall()
                    
                    if not columns:
                        return {"error": f"Table '{table_name}' not found"}
                    
                    return {
                        "table_name": table_name,
                        "columns": [dict(col) for col in columns]
                    }
        
        except Exception as e:
            logger.error(f"Schema retrieval error: {e}")
            return {"error": f"Error retrieving schema: {str(e)}"}
    
    async def list_tables(self) -> Dict[str, Any]:
        """List all tables in the database"""
        try:
            query = """
                SELECT table_name, table_type
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """
            
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query)
                    tables = cursor.fetchall()
                    return {"tables": [dict(table) for table in tables]}
        
        except Exception as e:
            logger.error(f"List tables error: {e}")
            return {"error": f"Error listing tables: {str(e)}"}
    
    async def get_all_schemas(self) -> Dict[str, List[str]]:
        """Get schema information for all tables"""
        try:
            tables_result = await self.list_tables()
            if "error" in tables_result:
                return {}
            
            schemas = {}
            for table in tables_result["tables"]:
                table_name = table["table_name"]
                schema_result = await self.get_table_schema(table_name)
                if "error" not in schema_result:
                    schemas[table_name] = [
                        f"{col['column_name']} ({col['data_type']})"
                        for col in schema_result["columns"]
                    ]
            
            return schemas
            
        except Exception as e:
            logger.error(f"Error getting all schemas: {e}")
            return {}

db_manager = DatabaseManager(db_config)

# Pydantic models with improved validation
class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=10000)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    conversation_history: Optional[List[ChatMessage]] = Field(default_factory=list)

class ChatResponse(BaseModel):
    response: str
    sql_query: Optional[str] = None
    tools_used: List[str] = Field(default_factory=list)
    execution_time: Optional[float] = None

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)

# Enhanced SQL query generation
class SQLGenerator:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.model = None
    
    def get_model(self):
        """Get or create Mistral model"""
        if not self.model:
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise HTTPException(status_code=500, detail="MISTRAL_API_KEY not configured")
            self.model = ChatMistralAI(
                model="mistral-large-2407",
                api_key=api_key,
                temperature=0.1  # Low temperature for more deterministic SQL generation
            )
        return self.model
    
    async def generate_sql(self, user_message: str, conversation_history: List[ChatMessage] = None) -> str:
        """Generate SQL query from natural language"""
        try:
            # Get schema information
            schemas = await self.db_manager.get_all_schemas()
            
            if not schemas:
                return "-- Error: Could not retrieve database schema"
            
            # Create schema description
            schema_text = "\n".join([
                f"Table {table}: {', '.join(columns)}"
                for table, columns in schemas.items()
            ])
            
            # Create enhanced prompt
            system_prompt = f"""You are an expert PostgreSQL query generator. Your task is to convert natural language requests into syntactically correct SELECT queries.

IMPORTANT RULES:
1. ONLY generate SELECT statements - never use DROP, DELETE, INSERT, UPDATE, ALTER, CREATE, TRUNCATE
2. Output ONLY the SQL query - no explanations, no markdown, no code blocks
3. Use proper PostgreSQL syntax
4. If the request cannot be fulfilled with a SELECT query, respond with: "-- Operation not allowed"
5. Use table and column names exactly as provided in the schema

DATABASE SCHEMA:
{schema_text}

Generate a SELECT query for the following request."""

            messages = [SystemMessage(content=system_prompt)]
            
            # Add conversation history if provided
            if conversation_history:
                for msg in conversation_history[-5:]:  # Last 5 messages for context
                    if msg.role == "user":
                        messages.append(HumanMessage(content=f"Previous request: {msg.content}"))
            
            messages.append(HumanMessage(content=f"Request: {user_message}"))
            
            model = self.get_model()
            response = await model.ainvoke(messages)
            
            print("=== RAW LLM OUTPUT ===")
            print(response.content)
            print("=== END RAW LLM OUTPUT ===")
            
            # Clean up the response
            sql_query = response.content.strip()
            
            # Remove any markdown code blocks
            sql_query = re.sub(r'```sql\s*', '', sql_query)
            sql_query = re.sub(r'```\s*', '', sql_query)
            
            # Remove common prefixes
            sql_query = re.sub(r'^(SQL|Query):\s*', '', sql_query, flags=re.IGNORECASE)
            
            return sql_query
            
        except Exception as e:
            logger.error(f"SQL generation error: {e}")
            return f"-- Error generating SQL: {str(e)}"

sql_generator = SQLGenerator(db_manager)

# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    try:
        logger.info("Starting application...")
        
        # Test database connection
        if await db_manager.test_connection():
            logger.info("Database connection successful")
        else:
            logger.warning("Database connection failed - some features may not work")
        
        # Initialize MCP (optional)
        await mcp_manager.initialize()
        
        logger.info("Application startup completed")
        yield
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        yield
    
    finally:
        logger.info("Shutting down application...")
        await mcp_manager.cleanup()
        logger.info("Application shutdown completed")

# FastAPI application
app = FastAPI(
    title="Restaurant Database Chatbot API",
    description="AI-powered chatbot for querying restaurant database",
    version="2.0.0",
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
    """Main chat endpoint for natural language to SQL conversion"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        logger.info(f"Processing chat request: {request.message[:100]}...")
        
        # Generate SQL query
        sql_query = await sql_generator.generate_sql(
            request.message, 
            request.conversation_history
        )
        
        if sql_query.startswith("-- Error") or sql_query.startswith("-- Operation not allowed"):
            return ChatResponse(
                response=sql_query,
                sql_query=sql_query,
                tools_used=[],
                execution_time=asyncio.get_event_loop().time() - start_time
            )
        
        # Execute the query
        result = await db_manager.execute_query(sql_query)
        
        if "error" in result:
            return ChatResponse(
                response=result["error"],
                sql_query=sql_query,
                tools_used=["sql_generator", "database"],
                execution_time=asyncio.get_event_loop().time() - start_time
            )
        else:
            # Return only the result as JSON string
            return ChatResponse(
                response=json.dumps(result, indent=2, default=str),
                sql_query=sql_query,
                tools_used=["sql_generator", "database"],
                execution_time=asyncio.get_event_loop().time() - start_time
            )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            response=f"An error occurred: {str(e)}",
            tools_used=["error_handler"],
            execution_time=asyncio.get_event_loop().time() - start_time
        )

@app.get("/health")
async def health_check():
    logger.info("/health endpoint called")
    try:
        # Test database
        db_healthy = await db_manager.test_connection()
        
        # Test Mistral API
        mistral_healthy = True
        try:
            sql_generator.get_model()
        except Exception:
            mistral_healthy = False
        
        # Overall status
        status = "healthy" if db_healthy and mistral_healthy else "degraded"
        
        return {
            "status": status,
            "timestamp": asyncio.get_event_loop().time(),
            "components": {
                "database": "healthy" if db_healthy else "unhealthy",
                "mistral_api": "healthy" if mistral_healthy else "unhealthy",
                "mcp_server": "optional"
            },
            "version": "2.0.0"
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": asyncio.get_event_loop().time()
        }

@app.get("/tables")
async def get_tables():
    logger.info("/tables endpoint called")
    try:
        tables_result = await db_manager.list_tables()
        if "error" in tables_result:
            raise HTTPException(status_code=500, detail=tables_result["error"])
        
        # Enrich with schema information
        enriched_tables = []
        for table in tables_result["tables"]:
            table_name = table["table_name"]
            schema_result = await db_manager.get_table_schema(table_name)
            
            enriched_table = {
                "name": table_name,
                "type": table["table_type"],
                "columns": schema_result.get("columns", []) if "error" not in schema_result else []
            }
            enriched_tables.append(enriched_table)
        
        return {"tables": enriched_tables}
        
    except Exception as e:
        logger.error(f"Error getting tables: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/table-schema/{table_name}")
async def get_table_schema_endpoint(table_name: str):
    logger.info(f"/table-schema/{table_name} endpoint called")
    try:
        result = await db_manager.get_table_schema(table_name)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Schema retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute-query")
async def execute_query_endpoint(request: QueryRequest):
    logger.info("/execute-query endpoint called")
    try:
        result = await db_manager.execute_query(request.query)
        return {"result": result, "query": request.query}
        
    except Exception as e:
        logger.error(f"Query execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def serve_frontend():
    return FileResponse("chatbot_frontend.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )