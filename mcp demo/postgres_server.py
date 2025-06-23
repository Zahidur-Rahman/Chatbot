import os
import json
import asyncio
import logging
import sys
from contextlib import contextmanager
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional
import re
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# MCP imports for stdio transport
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from pydantic import AnyUrl

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)  # Log to stderr to avoid interfering with stdio
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

logger.info(f"Starting postgres MCP server")
logger.info(f"Python executable: {sys.executable}")
logger.info(f"Current working directory: {os.getcwd()}")

# Create MCP server for stdio transport
server = Server("postgres-mcp-server")

class ChatResponse(BaseModel):
    result: Optional[dict] = None
    error: Optional[str] = None
    tools_used: List[str] = Field(default_factory=list)
    execution_time: Optional[float] = None

@contextmanager
def get_db_connection():
    """Get database connection with context manager"""
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            database=os.getenv("POSTGRES_DB", "resturent"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            port=os.getenv("POSTGRES_PORT", "5432")
        )
        logger.info("Database connection established")
        yield conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()
            logger.debug("Database connection closed")

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="execute_query",
            description="Execute a SQL SELECT query and return results",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL SELECT query to execute"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_table_schema",
            description="Get schema information for a specific table",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Name of the table to get schema for"
                    }
                },
                "required": ["table_name"]
            }
        ),
        Tool(
            name="list_tables",
            description="List all tables in the database",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_all_foreign_keys",
            description="Get all foreign key relationships in the database",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_full_schema",
            description="Get full schema: tables, columns, and foreign keys",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> ChatResponse:
    """Handle tool calls"""
    logger.info(f"Tool called: {name} with arguments: {arguments}")
    
    try:
        if name == "execute_query":
            result = await execute_query(arguments.get("query", ""))
            if "error" in result:
                return ChatResponse(
                    error=result["error"],
                    tools_used=["sql_generator", "database"],
                    execution_time=asyncio.get_event_loop().time() - start_time
                )
            else:
                return ChatResponse(
                    result=result,
                    tools_used=["sql_generator", "database"],
                    execution_time=asyncio.get_event_loop().time() - start_time
                )
        
        elif name == "get_table_schema":
            result = await get_table_schema(arguments.get("table_name", ""))
            if "error" in result:
                return ChatResponse(
                    error=result["error"],
                    tools_used=["sql_generator", "database"],
                    execution_time=asyncio.get_event_loop().time() - start_time
                )
            else:
                return ChatResponse(
                    result=result,
                    tools_used=["sql_generator", "database"],
                    execution_time=asyncio.get_event_loop().time() - start_time
                )
        
        elif name == "list_tables":
            result = await list_tables()
            if "error" in result:
                return ChatResponse(
                    error=result["error"],
                    tools_used=["sql_generator", "database"],
                    execution_time=asyncio.get_event_loop().time() - start_time
                )
            else:
                return ChatResponse(
                    result=result,
                    tools_used=["sql_generator", "database"],
                    execution_time=asyncio.get_event_loop().time() - start_time
                )
        
        elif name == "get_all_foreign_keys":
            result = await get_all_foreign_keys()
            if "error" in result:
                return ChatResponse(
                    error=result["error"],
                    tools_used=["sql_generator", "database"],
                    execution_time=asyncio.get_event_loop().time() - start_time
                )
            else:
                return ChatResponse(
                    result=result,
                    tools_used=["sql_generator", "database"],
                    execution_time=asyncio.get_event_loop().time() - start_time
                )
        
        elif name == "get_full_schema":
            result = await get_full_schema()
            if "error" in result:
                return ChatResponse(
                    error=result["error"],
                    tools_used=["sql_generator", "database"],
                    execution_time=asyncio.get_event_loop().time() - start_time
                )
            else:
                return ChatResponse(
                    result=result,
                    tools_used=["sql_generator", "database"],
                    execution_time=asyncio.get_event_loop().time() - start_time
                )
        
        else:
            error_msg = f"Unknown tool: {name}"
            logger.error(error_msg)
            return ChatResponse(
                error=error_msg,
                tools_used=["sql_generator", "database"],
                execution_time=asyncio.get_event_loop().time() - start_time
            )
    
    except Exception as e:
        error_msg = f"Error executing tool {name}: {str(e)}"
        logger.error(error_msg)
        return ChatResponse(
            error=error_msg,
            tools_used=["sql_generator", "database"],
            execution_time=asyncio.get_event_loop().time() - start_time
        )

async def execute_query(query: str) -> str:
    """Execute a SQL query and return results"""
    logger.info(f"Executing query: {query[:100]}...")  # Log first 100 chars
    
    try:
        # Basic input validation
        query = query.strip()
        if not query:
            return json.dumps({"error": "Empty query"})
        
        # Prevent dangerous operations (basic protection)
        dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']
        query_upper = query.upper()
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                error_msg = f"Operation not allowed: {keyword}"
                logger.warning(f"Blocked dangerous query: {query[:50]}...")
                return json.dumps({"error": error_msg})
        
        # Execute query
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query)
            
            if query_upper.startswith('SELECT'):
                results = cursor.fetchall()
                # Convert to regular dict for JSON serialization
                data = [dict(row) for row in results]
                response = {
                    "columns": list(data[0].keys()) if data else [],
                    "rows": [list(row.values()) for row in data],
                    "count": len(data)
                }
                logger.info(f"Query returned {len(data)} rows")
                return response
            else:
                conn.commit()
                response = {
                    "message": f"Query executed successfully. Rows affected: {cursor.rowcount}",
                    "rows_affected": cursor.rowcount
                }
                return response
                
    except psycopg2.Error as e:
        error_msg = f"Database error: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

async def get_table_schema(table_name: str) -> str:
    """Get schema information for a specific table"""
    logger.info(f"Getting schema for table: {table_name}")
    
    try:
        if not table_name:
            return json.dumps({"error": "Table name is required"})
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get column information
            query = """
                SELECT 
                    column_name, 
                    data_type, 
                    is_nullable, 
                    column_default,
                    character_maximum_length,
                    numeric_precision,
                    numeric_scale
                FROM information_schema.columns
                WHERE table_name = %s AND table_schema = 'public'
                ORDER BY ordinal_position;
            """
            cursor.execute(query, (table_name,))
            columns = cursor.fetchall()
            
            if not columns:
                return json.dumps({"error": f"Table '{table_name}' not found"})
            
            # Format schema information
            schema_info = {
                "table_name": table_name,
                "columns": []
            }
            
            for col in columns:
                column_info = {
                    "name": col[0],
                    "type": col[1],
                    "nullable": col[2] == 'YES',
                    "default": col[3],
                }
                
                # Add length/precision info if available
                if col[4]:  # character_maximum_length
                    column_info["max_length"] = col[4]
                if col[5]:  # numeric_precision
                    column_info["precision"] = col[5]
                if col[6]:  # numeric_scale
                    column_info["scale"] = col[6]
                
                schema_info["columns"].append(column_info)
            
            logger.info(f"Schema retrieved for table {table_name}: {len(columns)} columns")
            return schema_info
            
    except psycopg2.Error as e:
        error_msg = f"Database error getting schema: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})
    except Exception as e:
        error_msg = f"Unexpected error getting schema: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

async def list_tables() -> str:
    """List all tables in the database"""
    logger.info("Listing tables")
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get table names and row counts
            query = """
                SELECT 
                    t.table_name,
                    COALESCE(s.n_tup_ins - s.n_tup_del, 0) as estimated_rows
                FROM information_schema.tables t
                LEFT JOIN pg_stat_user_tables s ON s.relname = t.table_name
                WHERE t.table_schema = 'public' AND t.table_type = 'BASE TABLE'
                ORDER BY t.table_name;
            """
            cursor.execute(query)
            tables = cursor.fetchall()
            
            table_info = {
                "tables": [
                    {
                        "name": table[0],
                        "estimated_rows": table[1] if table[1] is not None else 0
                    }
                    for table in tables
                ],
                "count": len(tables)
            }
            
            logger.info(f"Listed {len(tables)} tables")
            return table_info
            
    except psycopg2.Error as e:
        error_msg = f"Database error listing tables: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})
    except Exception as e:
        error_msg = f"Unexpected error listing tables: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

async def get_all_foreign_keys() -> str:
    """Get all foreign key relationships in the database"""
    logger.info("Getting all foreign keys")
    query = """
        SELECT
            tc.table_name AS table_name,
            kcu.column_name AS column_name,
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name
        FROM
            information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name
              AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
              ON ccu.constraint_name = tc.constraint_name
              AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public';
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            fks = cursor.fetchall()
    return json.dumps({"foreign_keys": fks}, indent=2)

async def get_full_schema() -> str:
    """Get full schema: tables, columns, and foreign keys"""
    logger.info("Getting full schema")
    # Get tables and columns
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """)
        tables = [row['table_name'] for row in cursor.fetchall()]
        schema = {}
        for table in tables:
            cursor.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = %s AND table_schema = 'public'
                ORDER BY ordinal_position;
            """, (table,))
            schema[table] = cursor.fetchall()
    # Get foreign keys
    fks = json.loads(await get_all_foreign_keys())['foreign_keys']
    return json.dumps({"tables": schema, "foreign_keys": fks}, indent=2)

async def main():
    """Main function to run the MCP server"""
    logger.info("Starting MCP server with stdio transport...")
    
    try:
        # Test database connection on startup
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            logger.info(f"Connected to PostgreSQL: {version}")
        
        # Run the server
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
            
    except KeyboardInterrupt:
        logger.info("MCP server stopped by user")
    except Exception as e:
        logger.error(f"MCP server error: {e}")
        raise
    finally:
        logger.info("MCP server stopped")

if __name__ == "__main__":
    asyncio.run(main())