import dspy
import yaml
import json
import os
import re
import chardet
from typing import Tuple
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM('openai/gpt-4.1-nano', api_key=os.getenv('OPENAI_API_KEY'))
dspy.configure(lm=lm)


def safe_read_file(file_path: str) -> str:
    """
    Safely read a file with automatic encoding detection.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                detected = chardet.detect(raw_data)
                encoding = detected['encoding'] or 'latin-1'
            
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

class TaskAnalysis(dspy.Signature):
    """Analyze task.yaml to extract UI requirements and API specifications."""
    task_yaml_content: str = dspy.InputField(desc="Content of task.yaml")
    task_type: str = dspy.OutputField(desc="Task type (e.g., object_detection)")
    ui_requirements: str = dspy.OutputField(desc="Summary of UI requirements")
    input_components: str = dspy.OutputField(desc="Comma-separated input components (e.g., input_file, input_text)")
    output_components: str = dspy.OutputField(desc="Comma-separated output components (e.g., output_image, output_text)")
    input_payload: str = dspy.OutputField(desc="Expected input payload structure for Python backend")
    output_payload: str = dspy.OutputField(desc="Expected output payload structure from Python backend")
    input_type: str = dspy.OutputField(desc="Primary input type (image, text, audio)")
    output_type: str = dspy.OutputField(desc="Primary output type (image, text, json)")

class UIComponentGeneration(dspy.Signature):
    """Generate UI components with Tailwind styling based on task requirements."""
    task_type: str = dspy.InputField()
    component_type: str = dspy.InputField(desc="Component type (e.g., input_file)")
    requirements: str = dspy.InputField(desc="Specific requirements for this component")
    model_info: str = dspy.InputField(desc="Model API specifications")
    component_code: str = dspy.OutputField(desc="HTML/JS code for the component")

class APIIntegration(dspy.Signature):
    """Generate frontend API integration code to communicate with Python backend ONLY."""
    input_components: str = dspy.InputField(desc="Comma-separated input component types")
    output_components: str = dspy.InputField(desc="Comma-separated output component types")
    input_payload: str = dspy.InputField(desc="Expected input payload structure for Python backend")
    output_payload: str = dspy.InputField(desc="Expected output payload structure from Python backend")
    task_type: str = dspy.InputField()
    integration_code: str = dspy.OutputField(desc="JavaScript API integration code for Python backend")

class UILayoutGeneration(dspy.Signature):
    """Generate complete UI layout with responsive design."""
    task_name: str = dspy.InputField()
    task_description: str = dspy.InputField()
    input_components: list = dspy.InputField(desc="List of HTML strings for input components")
    output_components: list = dspy.InputField(desc="List of HTML strings for output components")
    api_integration: str = dspy.InputField(desc="JS integration code for Python backend")
    complete_html: str = dspy.OutputField(desc="Full HTML page with styling")

class BackendGeneration(dspy.Signature):
    """Generate Python Flask backend to interface with model server."""
    input_payload: str = dspy.InputField(desc="Expected input payload from frontend")
    output_payload: str = dspy.InputField(desc="Expected output payload to frontend")
    task_type: str = dspy.InputField()
    model_info: str = dspy.InputField(desc="Model API specifications")
    backend_code: str = dspy.OutputField(desc="Python Flask backend code")

class UIGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyze_task = dspy.ChainOfThought(TaskAnalysis)
        self.generate_component = dspy.ChainOfThought(UIComponentGeneration)
        self.generate_api_integration = dspy.ChainOfThought(APIIntegration)
        self.generate_layout = dspy.ChainOfThought(UILayoutGeneration)
        self.generate_backend = dspy.ChainOfThought(BackendGeneration)
    
    def forward(self, task_yaml_path: str):
        # Read task.yaml with robust encoding handling
        task_yaml_content = safe_read_file(task_yaml_path)
        
        try:
            task_data = yaml.safe_load(task_yaml_content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML content in {task_yaml_path}: {e}")
        
        model_info = task_data.get('model_information', {})

        # Step 1: Analyze task requirements
        analysis = self.analyze_task(task_yaml_content=task_yaml_content)
        
        # Parse component lists
        input_comp_list = [c.strip() for c in analysis.input_components.split(',')]
        output_comp_list = [c.strip() for c in analysis.output_components.split(',')]

        # Step 2: Generate backend code
        backend = self.generate_backend(
            task_type=analysis.task_type,
            input_payload=analysis.input_payload,
            output_payload=analysis.output_payload,
            model_info=json.dumps(model_info)
        )

        try:
            input_payload = safe_read_file("curl_command_generated.txt")
            output_payload = safe_read_file("response.json")
            print("Successfully loaded curl_command_generated.txt and response.json")
        except FileNotFoundError as e:
            print(f"Warning: {e}. Using payload from analysis.")
            input_payload = analysis.input_payload
            output_payload = analysis.output_payload

        # Step 3: Generate API integration (frontend -> Python backend)
        api_integration = self.generate_api_integration(
            task_type=analysis.task_type,
            input_components=analysis.input_components,
            output_components=analysis.output_components,
            input_payload = input_payload[:1000],
            output_payload=output_payload
        )

        # Step 4: Generate input components
        input_components_html = []
        for comp_type in input_comp_list:
            comp = self.generate_component( 
                task_type=analysis.task_type,
                component_type=comp_type,
                requirements=f"{comp_type} for {analysis.task_type}",
                model_info=json.dumps(model_info)
            )
            input_components_html.append(comp.component_code)
        
        # Step 5: Generate output components
        output_components_html = []
        for comp_type in output_comp_list:
            comp = self.generate_component(
                task_type=analysis.task_type,
                component_type=comp_type,
                requirements=f"{comp_type} for {analysis.task_type}",
                model_info=json.dumps(model_info))
            output_components_html.append(comp.component_code)
        
        # Step 6: Generate complete UI
        complete_ui = self.generate_layout(
            task_name=task_data.get('task_description', {}).get('type', 'ML Task'),
            task_description=task_data.get('task_description', {}).get('description', ''),
            input_components=input_components_html,
            output_components=output_components_html,
            api_integration=api_integration.integration_code
        )
        
        return dspy.Prediction(
            ui_html=complete_ui.complete_html,
            backend_code=backend.backend_code,
            analysis=analysis
        )

class AutoUIGenerator:
    def __init__(self):
        self.ui_generator = UIGenerator()
    
    def generate(self, task_problem_dir: str) -> Tuple[str, str]:
        task_yaml_path = os.path.join(task_problem_dir, 'task.yaml')
        if not os.path.exists(task_yaml_path):
            raise FileNotFoundError(f"task.yaml not found in {task_problem_dir}")
        
        result = self.ui_generator(task_yaml_path=task_yaml_path)
        return result.ui_html, result.backend_code
    
    def save(self, ui_html: str, backend_code: str, output_dir: str, task_name: str):
        os.makedirs(output_dir, exist_ok=True)
        
        def clean_code(code):
            return re.sub(r'```(html|python)?\n|\n```', '', code).strip()
        
        ui_path = os.path.join(output_dir, f"{task_name}_ui.html")
        with open(ui_path, 'w', encoding='utf-8-sig') as f:
            f.write(clean_code(ui_html))
        
        backend_path = os.path.join(output_dir, f"{task_name}_backend.py")
        with open(backend_path, 'w', encoding='utf-8-sig') as f:
            f.write(clean_code(backend_code))
        
        return ui_path, backend_path

if __name__ == "__main__":
    generator = AutoUIGenerator()
    task_name = "object_detection_in_image"
    task_problem_dir = f"../problems/{task_name}"
    
    try:
        print(f"Generating UI for task: {task_name}")
        print(f"Looking for task.yaml in: {task_problem_dir}")
        
        ui_html, backend_code = generator.generate(task_problem_dir)
        ui_path, backend_path = generator.save(ui_html, backend_code, "../results", task_name)
        
        print("UI and Backend generated successfully!")
        print(f"UI saved to: {ui_path}")
        print(f"Backend saved to: {backend_path}")
        
        # Print backend usage instructions
        print("\nHOW TO USE THE BACKEND:")
        print("1. Install requirements: pip install flask requests python-dotenv chardet")
        print("2. Create a .env file with your API keys")
        print(f"3. Run the backend: python results/{task_name}_backend.py")
        print(f"4. Open the UI in browser: results/{task_name}_ui.html")
        
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()