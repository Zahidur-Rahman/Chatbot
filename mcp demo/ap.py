#!/usr/bin/env python3
"""
Windows-compatible MCP Server for PostgreSQL database operations.
This server provides tools for executing SQL queries and managing database schema.
"""

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# MCP imports
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    TextContent,
    Tool,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("postgres-mcp-server")

# Database configuration
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "database": os.getenv("POSTGRES_DB", "resturent"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
    "port": int(os.getenv("POSTGRES_PORT", "5432"))
}

class PostgreSQLServer:
    """PostgreSQL MCP Server implementation"""
    
    def __init__(self):
        self.server = Server("postgres-server")
        self._setup_handlers()
    
    def get_db_connection(self):
        """Get database connection (sync version for Windows compatibility)"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            return conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def _setup_handlers(self):
        """Setup MCP request handlers"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="execute_query",
                    description="Execute a SQL query and return results. Only SELECT queries are allowed for security.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The SQL query to execute (SELECT statements only)"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_table_schema",
                    description="Get detailed schema information for a specific table",
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
                    description="List all tables in the public schema",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls"""
            try:
                if name == "execute_query":
                    result = await self._execute_query(arguments.get("query", ""))
                elif name == "get_table_schema":
                    result = await self._get_table_schema(arguments.get("table_name", ""))
                elif name == "list_tables":
                    result = await self._list_tables()
                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
                
                return [TextContent(type="text", text=result)]
                
            except Exception as e:
                logger.error(f"Tool call error for {name}: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def _execute_query(self, query: str) -> str:
        """Execute a SQL query with security checks"""
        try:
            # Input validation
            query = query.strip()
            if not query:
                return json.dumps({"error": "Empty query provided"})
            
            # Security check - only allow SELECT statements
            if not query.upper().startswith('SELECT'):
                return json.dumps({
                    "error": "Only SELECT queries are allowed for security reasons",
                    "provided_query": query[:100] + "..." if len(query) > 100 else query
                })
            
            # Check for dangerous patterns
            dangerous_patterns = [
                'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE',
                'GRANT', 'REVOKE', 'COPY', 'CALL', 'EXECUTE'
            ]
            
            query_upper = query.upper()
            for pattern in dangerous_patterns:
                if pattern in query_upper:
                    return json.dumps({
                        "error": f"Query contains forbidden keyword: {pattern}",
                        "provided_query": query[:100] + "..." if len(query) > 100 else query
                    })
            
            # Execute query synchronously for Windows compatibility
            conn = self.get_db_connection()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query)
                    results = cursor.fetchall()
                    
                    # Convert to JSON-serializable format
                    json_results = []
                    for row in results:
                        json_row = {}
                        for key, value in row.items():
                            # Handle special types that aren't JSON serializable
                            if hasattr(value, 'isoformat'):  # datetime objects
                                json_row[key] = value.isoformat()
                            elif isinstance(value, (bytes, memoryview)):
                                json_row[key] = str(value)
                            else:
                                json_row[key] = value
                        json_results.append(json_row)
                    
                    return json.dumps({
                        "success": True,
                        "query": query,
                        "row_count": len(json_results),
                        "results": json_results
                    }, indent=2)
            finally:
                conn.close()
                    
        except psycopg2.Error as e:
            logger.error(f"Database error: {e}")
            return json.dumps({
                "error": f"Database error: {str(e)}",
                "query": query[:100] + "..." if len(query) > 100 else query
            })
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return json.dumps({
                "error": f"Unexpected error: {str(e)}",
                "query": query[:100] + "..." if len(query) > 100 else query
            })
    
    async def _get_table_schema(self, table_name: str) -> str:
        """Get detailed schema information for a table"""
        try:
            if not table_name:
                return json.dumps({"error": "Table name is required"})
            
            conn = self.get_db_connection()
            try:
                with conn.cursor() as cursor:
                    # Get column information
                    cursor.execute("""
                        SELECT 
                            column_name,
                            data_type,
                            character_maximum_length,
                            is_nullable,
                            column_default,
                            ordinal_position
                        FROM information_schema.columns
                        WHERE table_name = %s AND table_schema = 'public'
                        ORDER BY ordinal_position;
                    """, (table_name,))
                    
                    columns = cursor.fetchall()
                    if not columns:
                        return json.dumps({"error": f"Table '{table_name}' not found"})
                    
                    schema_info = []
                    for col in columns:
                        col_info = {
                            "name": col[0],
                            "type": col[1],
                            "max_length": col[2],
                            "nullable": col[3] == "YES",
                            "default": col[4],
                            "position": col[5]
                        }
                        schema_info.append(col_info)
                    
                    return json.dumps({
                        "table_name": table_name,
                        "columns": schema_info,
                        "column_count": len(schema_info)
                    }, indent=2)
            finally:
                conn.close()
                    
        except Exception as e:
            logger.error(f"Schema error: {e}")
            return json.dumps({"error": f"Error getting schema: {str(e)}"})
    
    async def _list_tables(self) -> str:
        """List all tables in the public schema"""
        try:
            conn = self.get_db_connection()
            try:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT 
                            table_name,
                            table_type
                        FROM information_schema.tables 
                        WHERE table_schema = 'public'
                        ORDER BY table_name;
                    """)
                    
                    tables = cursor.fetchall()
                    table_list = [{"name": table[0], "type": table[1]} for table in tables]
                    
                    return json.dumps({
                        "tables": table_list,
                        "count": len(table_list)
                    }, indent=2)
            finally:
                conn.close()
                    
        except Exception as e:
            logger.error(f"List tables error: {e}")
            return json.dumps({"error": f"Error listing tables: {str(e)}"})

async def main():
    """Main function to run the MCP server"""
    try:
        # Test database connection
        conn = psycopg2.connect(**DB_CONFIG)
        conn.close()
        logger.info("Database connection test successful")
        
        # Create server
        postgres_server = PostgreSQLServer()
        
        # Run the MCP server with stdio transport
        async with stdio_server() as (read_stream, write_stream):
            logger.info("Starting PostgreSQL MCP server...")
            await postgres_server.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="postgres-server",
                    server_version="1.0.0",
                    capabilities=postgres_server.server.get_capabilities(
                        notification_options=None,
                        experimental_capabilities=None,
                    ),
                ),
            )
            
    except psycopg2.Error as e:
        logger.error(f"Database connection failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Set event loop policy for Windows compatibility
    logger.info("Start checking OS")
    if sys.platform.startswith('win'):
        logger.info("Get Windows")
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        logger.info("Windows Operations Done")
    
    asyncio.run(main())