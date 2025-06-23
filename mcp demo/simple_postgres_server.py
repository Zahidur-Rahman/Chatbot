import asyncio
import json
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        database=os.getenv("POSTGRES_DB", "resturent"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        port=os.getenv("POSTGRES_PORT", "5432")
    )

async def execute_query(query: str) -> str:
    """Execute a SQL query and return results"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query)
        
        if query.strip().upper().startswith('SELECT'):
            results = cursor.fetchall()
            return str([dict(row) for row in results])
        else:
            conn.commit()
            return f"Query executed successfully. Rows affected: {cursor.rowcount}"
    except Exception as e:
        return f"Error executing query: {str(e)}"
    finally:
        if 'conn' in locals():
            conn.close()

async def get_table_schema(table_name: str) -> str:
    """Get schema information for a specific table"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = %s
        ORDER BY ordinal_position;
        """
        
        cursor.execute(query, (table_name,))
        columns = cursor.fetchall()
        
        schema_info = f"Schema for table '{table_name}':\n"
        for col in columns:
            schema_info += f"- {col[0]}: {col[1]} (nullable: {col[2]}, default: {col[3]})\n"
        
        return schema_info
    except Exception as e:
        return f"Error getting schema: {str(e)}"
    finally:
        if 'conn' in locals():
            conn.close()

async def list_tables() -> str:
    """List all tables in the database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
        """
        
        cursor.execute(query)
        tables = cursor.fetchall()
        
        return f"Tables in database: {[table[0] for table in tables]}"
    except Exception as e:
        return f"Error listing tables: {str(e)}"
    finally:
        if 'conn' in locals():
            conn.close()

# MCP Protocol Implementation
async def handle_mcp_protocol():
    """Handle MCP protocol communication"""
    while True:
        try:
            # Read input
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            
            data = json.loads(line.strip())
            
            if data.get("method") == "initialize":
                # Send initialization response
                response = {
                    "jsonrpc": "2.0",
                    "id": data.get("id"),
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "postgres-server",
                            "version": "1.0.0"
                        }
                    }
                }
                print(json.dumps(response))
                print(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}))
                
            elif data.get("method") == "tools/list":
                # List available tools
                response = {
                    "jsonrpc": "2.0",
                    "id": data.get("id"),
                    "result": {
                        "tools": [
                            {
                                "name": "execute_query",
                                "description": "Execute a SQL query and return results",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string", "description": "SQL query to execute"}
                                    },
                                    "required": ["query"]
                                }
                            },
                            {
                                "name": "get_table_schema",
                                "description": "Get schema information for a specific table",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "table_name": {"type": "string", "description": "Name of the table"}
                                    },
                                    "required": ["table_name"]
                                }
                            },
                            {
                                "name": "list_tables",
                                "description": "List all tables in the database",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {}
                                }
                            }
                        ]
                    }
                }
                print(json.dumps(response))
                
            elif data.get("method") == "tools/call":
                # Handle tool calls
                params = data.get("params", {})
                name = params.get("name")
                arguments = params.get("arguments", {})
                
                result = ""
                if name == "execute_query":
                    result = await execute_query(arguments.get("query", ""))
                elif name == "get_table_schema":
                    result = await get_table_schema(arguments.get("table_name", ""))
                elif name == "list_tables":
                    result = await list_tables()
                else:
                    result = f"Unknown tool: {name}"
                
                response = {
                    "jsonrpc": "2.0",
                    "id": data.get("id"),
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": result
                            }
                        ]
                    }
                }
                print(json.dumps(response))
                
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": data.get("id") if 'data' in locals() else None,
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }
            print(json.dumps(error_response))

if __name__ == "__main__":
    asyncio.run(handle_mcp_protocol()) 