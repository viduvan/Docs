"""
Export all workflows from tradeflat n8n PostgreSQL database as individual JSON files.
Each file follows the n8n workflow export format.

Usage:
    python3 export_workflows.py
    
Requires: psycopg2-binary (pip install psycopg2-binary)
"""
import json
import os
import re
import psycopg2

# Connect to the temporary postgres container on port 3099
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3099,
    "database": "n8n",
    "user": "postgres",
    "password": "123456aA@"
}

OUTPUT_DIR = "/home/vietpv/FIS/tradeflat/n8n/database/tradeflat_export"

def sanitize_filename(name: str) -> str:
    """Convert workflow name to a safe filename."""
    # Replace spaces and special chars with underscores
    safe = re.sub(r'[^\w\s\-.]', '', name)
    safe = re.sub(r'\s+', '_', safe.strip())
    return safe[:100]  # limit length


def export_workflows():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Query all workflows with all relevant fields
    cur.execute("""
        SELECT id, name, active, nodes, connections, settings, "staticData", "pinData", meta, "triggerCount"
        FROM workflow_entity
        ORDER BY id
    """)
    
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    
    all_workflows = []
    
    for row in rows:
        wf = dict(zip(columns, row))
        
        # Build n8n-compatible workflow JSON
        workflow_json = {
            "id": wf["id"],
            "name": wf["name"],
            "active": wf["active"],
            "nodes": wf["nodes"] if isinstance(wf["nodes"], (list, dict)) else json.loads(wf["nodes"]) if wf["nodes"] else [],
            "connections": wf["connections"] if isinstance(wf["connections"], dict) else json.loads(wf["connections"]) if wf["connections"] else {},
            "settings": wf["settings"] if isinstance(wf["settings"], dict) else json.loads(wf["settings"]) if wf["settings"] else {},
            "staticData": wf["staticData"] if isinstance(wf["staticData"], (dict, type(None))) else json.loads(wf["staticData"]) if wf["staticData"] else None,
            "pinData": wf["pinData"] if isinstance(wf["pinData"], (dict, type(None))) else json.loads(wf["pinData"]) if wf["pinData"] else None,
            "meta": wf["meta"] if isinstance(wf["meta"], (dict, type(None))) else json.loads(wf["meta"]) if wf["meta"] else None,
            "triggerCount": wf["triggerCount"],
        }
        
        all_workflows.append(workflow_json)
        
        # Save individual workflow file
        filename = f"{sanitize_filename(wf['name'])}_{wf['id']}.json"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(workflow_json, f, ensure_ascii=False, indent=2, default=str)
        
        status = "✅ active" if wf["active"] else "⬜ inactive"
        print(f"  {status} | {wf['name']} -> {filename}")
    
    # Save combined all_workflows.json
    all_file = os.path.join(OUTPUT_DIR, "_all_workflows.json")
    with open(all_file, "w", encoding="utf-8") as f:
        json.dump(all_workflows, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Exported {len(rows)} workflows to: {OUTPUT_DIR}")
    print(f"Combined file: {all_file}")
    
    cur.close()
    conn.close()


if __name__ == "__main__":
    export_workflows()
