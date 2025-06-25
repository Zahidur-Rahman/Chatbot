import os
import json
import asyncio
import logging
import sys
from contextlib import contextmanager
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional, Any
import re
from datetime import datetime

# MCP imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Set up logging for Windows compatibility
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

logger.info("Starting postgres MCP server")
logger.info(f"Python executable: {sys.executable}")
logger.info(f"Current working directory: {os.getcwd()}")

# Create MCP server
server = Server("postgres-mcp-server")

@contextmanager
def get_db_connection():
    """Get database connection with context manager"""
    conn = None
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            database=os.getenv("POSTGRES_DB", "resturent"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            port=int(os.getenv("POSTGRES_PORT", "5432"))
        )
        logger.debug("Database connection established")
        yield conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logger.debug("Database connection closed")

def validate_query(query: str) -> tuple[bool, str]:
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

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools - matching FastMCP client expectations"""
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
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    """Handle tool calls and return MCP-compliant responses"""
    logger.info(f"Tool called: {name} with arguments: {arguments}")
    
    # Handle None or non-dict arguments
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        logger.error(f"Arguments is not a dict! Type: {type(arguments)}, Value: {arguments}")
        return [TextContent(type="text", text=json.dumps({"error": "Arguments must be a dict."}))]
    
    try:
        if name == "execute_query":
            result = await execute_query(arguments.get("query", ""))
        elif name == "get_table_schema":
            result = await get_table_schema(arguments.get("table_name", ""))
        elif name == "list_tables":
            result = await list_tables()
        elif name == "get_full_schema":
            result = await get_full_schema()
        else:
            result = {"error": f"Unknown tool: {name}"}
        
        # Return MCP-compliant response
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
    except Exception as e:
        error_msg = f"Error executing tool {name}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return [TextContent(type="text", text=json.dumps({"error": error_msg}))]

async def execute_query(query: str) -> Dict[str, Any]:
    """Execute a SQL query and return results - matching FastMCP client expectations"""
    logger.info(f"Executing query: {query[:100]}...")
    
    try:
        # Validate query
        is_valid, message = validate_query(query)
        if not is_valid:
            return {"error": message}
        
        # Execute query
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query)
                results = cursor.fetchall()
                
                # Convert to serializable format
                data = [dict(row) for row in results]
                
                # Match the expected format from FastMCP client
                response = {
                    "columns": list(data[0].keys()) if data else [],
                    "rows": data,
                    "row_count": len(data),  # FastMCP client expects 'row_count'
                    "query": query
                }
                
                logger.info(f"Query returned {len(data)} rows")
                return response
                
    except psycopg2.Error as e:
        error_msg = f"Database error: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}

async def get_table_schema(table_name: str) -> Dict[str, Any]:
    """Get schema information for a specific table - matching FastMCP client expectations"""
    logger.info(f"Getting schema for table: {table_name}")
    
    try:
        if not table_name:
            return {"error": "Table name is required"}
        
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get column information
                query = """
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
                """
                cursor.execute(query, (table_name,))
                columns = cursor.fetchall()
                
                if not columns:
                    return {"error": f"Table '{table_name}' not found"}
                
                # Get primary keys
                pk_query = """
                    SELECT kcu.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu 
                        ON tc.constraint_name = kcu.constraint_name
                    WHERE tc.table_name = %s 
                        AND tc.constraint_type = 'PRIMARY KEY'
                        AND tc.table_schema = 'public';
                """
                cursor.execute(pk_query, (table_name,))
                primary_keys = [row['column_name'] for row in cursor.fetchall()]
                
                # Format to match FastMCP client expectations
                response = {
                    "table_name": table_name,
                    "columns": [dict(col) for col in columns],
                    "primary_keys": primary_keys
                }
                
                logger.info(f"Schema retrieved for table {table_name}: {len(columns)} columns")
                return response
                
    except psycopg2.Error as e:
        error_msg = f"Database error getting schema: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error getting schema: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}

async def list_tables() -> Dict[str, Any]:
    """List all tables in the database - matching FastMCP client expectations"""
    logger.info("Listing tables")
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get table names and row counts - matching the query from FastMCP client
                query = """
                    SELECT 
                        t.table_name,
                        COALESCE(s.n_tup_ins - s.n_tup_del, 0) as estimated_rows,
                        obj_description(c.oid) as table_comment
                    FROM information_schema.tables t
                    LEFT JOIN pg_stat_user_tables s ON s.relname = t.table_name
                    LEFT JOIN pg_class c ON c.relname = t.table_name
                    WHERE t.table_schema = 'public' AND t.table_type = 'BASE TABLE'
                    ORDER BY t.table_name;
                """
                cursor.execute(query)
                tables = cursor.fetchall()
                
                # Match the expected format from FastMCP client
                response = {
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
                
                logger.info(f"Listed {len(tables)} tables")
                return response
                
    except psycopg2.Error as e:
        error_msg = f"Database error listing tables: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error listing tables: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}

async def get_full_schema() -> Dict[str, Any]:
    """Get full schema: tables, columns, and foreign keys - matching FastMCP client expectations"""
    logger.info("Getting full schema")
    
    try:
        # Get tables first
        tables_result = await list_tables()
        if "error" in tables_result:
            return tables_result
        
        # Initialize the schema structure to match FastMCP client expectations
        full_schema = {"tables": {}}
        
        # Get detailed schema for each table
        for table_info in tables_result["tables"]:
            table_name = table_info["name"]
            schema_result = await get_table_schema(table_name)
            
            if "error" not in schema_result:
                # Store the columns data as expected by FastMCP client
                full_schema["tables"][table_name] = schema_result["columns"]
        
        logger.info(f"Full schema retrieved with {len(full_schema['tables'])} tables")
        return full_schema
        
    except Exception as e:
        error_msg = f"Error getting full schema: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}

async def test_database_connection():
    """Test database connection on startup"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT version();")
                version = cursor.fetchone()[0]
                logger.info(f"✓ Connected to PostgreSQL: {version}")
                
                # Test basic functionality
                cursor.execute("""
                    SELECT COUNT(*) as table_count 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                """)
                table_count = cursor.fetchone()[0]
                logger.info(f"✓ Found {table_count} tables in public schema")
                return True
    except Exception as e:
        logger.error(f"✗ Database connection test failed: {e}")
        return False

async def main():
    """Main function to run the MCP server"""
    logger.info("Starting PostgreSQL MCP server with stdio transport...")
    
    try:
        # Test database connection on startup
        if not await test_database_connection():
            logger.error("Database connection test failed. Please check your configuration.")
            sys.exit(1)
        
        logger.info("✓ All systems ready - starting MCP server")
        
        # Run the server with stdio transport
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
            
    except KeyboardInterrupt:
        logger.info("MCP server interrupted by user")
    except Exception as e:
        logger.error(f"MCP server error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("MCP server stopped")

if __name__ == "__main__":
    # Windows compatibility: handle event loop properly
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}", exc_info=True)
        sys.exit(1)