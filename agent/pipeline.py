import asyncio
import os
import sys
import traceback
from agent_curl_generator import generate_curl_for_task
from agent_execute_curl import execute_curl_for_task
from agent_dspy_v8 import AutoUIGenerator

async def process_task(task_name: str):
    """Process a single task through all stages"""
    print(f"\n{'='*50}")
    print(f"🚀 Starting processing for task: {task_name}")
    print(f"{'='*50}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    task_dir = os.path.join(base_dir, "problems", task_name)
    
    # Stage 1: Generate curl command
    print("\n🔧 Stage 1: Generating curl command")
    try:
        curl_command = await generate_curl_for_task(task_name)
        print(f"  ✅ Curl command generated for {task_name}")
    except Exception as e:
        print(f"  ❌ Failed to generate curl command: {str(e)}")
        return False
    
    # Stage 2: Execute curl command
    print("\n⚡ Stage 2: Executing curl command")
    try:
        success = execute_curl_for_task(task_name)
        if success:
            print(f"  ✅ Response saved for {task_name}")
        else:
            print(f"  ❌ Failed to execute curl command")
            return False
    except Exception as e:
        print(f"  ❌ Error executing curl: {str(e)}")
        return False
    
    # Stage 3: Generate UI
    print("\n🎨 Stage 3: Generating UI")
    try:
        print("  📂 Loading task configuration...")
        generator = AutoUIGenerator()
        ui_html = generator.generate(task_dir)
        
        print("  💾 Saving UI...")
        ui_path = generator.save(ui_html, task_dir, task_name)
        
        print(f"  ✅ UI generated successfully! Saved to: {ui_path}")
        return True
    except Exception as e:
        print(f"  ❌ Error generating UI: {str(e)}")
        print(f"  🔍 Error type: {type(e).__name__}")
        traceback.print_exc()
        return False

async def main():
    # Get base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    problems_dir = os.path.join(base_dir, "problems")
    
    # Get all task folders
    task_folders = [f for f in os.listdir(problems_dir) 
                   if os.path.isdir(os.path.join(problems_dir, f))]
    
    # Process specific task if provided as argument
    if len(sys.argv) > 1:
        task_name = sys.argv[1]
        if task_name in task_folders:
            await process_task(task_name)
        else:
            print(f"❌ Task not found: {task_name}")
            sys.exit(1)
    else:
        # Process all tasks
        for task_name in task_folders:
            await process_task(task_name)

if __name__ == "__main__":
    asyncio.run(main())