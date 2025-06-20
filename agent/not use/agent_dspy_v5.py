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
    input_payload: str = dspy.OutputField(desc="Expected input payload structure for model server")
    output_payload: str = dspy.OutputField(desc="Expected output payload structure from model server")
    input_type: str = dspy.OutputField(desc="Primary input type (image, text, audio)")
    output_type: str = dspy.OutputField(desc="Primary output type (image, text, json)")

class UIComponentGeneration(dspy.Signature):
    """Generate UI components with Tailwind styling based on task requirements."""
    task_type: str = dspy.InputField()
    component_type: str = dspy.InputField(desc="Component type (e.g., input_file)")
    requirements: str = dspy.InputField(desc="Specific requirements for this component")
    model_info: str = dspy.InputField(desc="Model API specifications")
    input_type: str = dspy.InputField(desc="Primary input data modality (e.g., audio, image, text).")
    output_type: str = dspy.InputField(desc="Primary output data modality (e.g., json, label, text, image).")
    component_code: str = dspy.OutputField(desc="HTML/JS code for the component")

class APIIntegration(dspy.Signature):
    """Generate frontend API integration code to communicate directly with model server."""
    input_components: str = dspy.InputField(desc="Comma-separated input component types")
    output_components: str = dspy.InputField(desc="Comma-separated output component types")
    input_payload: str = dspy.InputField(desc="Expected input payload structure for model server")
    output_payload: str = dspy.InputField(desc="Expected output payload structure from model server")
    task_type: str = dspy.InputField()
    model_info: str = dspy.InputField(desc="Model API specifications including api_url")
    input_type: str = dspy.InputField(desc="Primary input modality (audio, image, text, tabular).")
    output_type: str = dspy.InputField(desc="Primary output modality (json, image, text).")
    integration_code: str = dspy.OutputField(desc="JavaScript API integration code for model server. Use data from response.data")

class UILayoutGeneration(dspy.Signature):
    """Generate a complete, valid, and responsive HTML5 layout that includes a <html> tag, <head>, <body>, a header, footer, and main content section. Ensure it follows proper HTML syntax and formatting."""
    task_name: str = dspy.InputField()
    task_description: str = dspy.InputField()
    input_components: list = dspy.InputField(desc="List of HTML strings for input components")
    output_components: list = dspy.InputField(desc="List of HTML strings for output components")
    api_integration: str = dspy.InputField(desc="JS integration code for model server")
    complete_html: str = dspy.OutputField(desc="Full HTML page with styling")

class UIGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyze_task = dspy.ChainOfThought(TaskAnalysis)
        self.generate_component = dspy.ChainOfThought(UIComponentGeneration)
        self.generate_api_integration = dspy.ChainOfThought(APIIntegration)
        self.generate_layout = dspy.ChainOfThought(UILayoutGeneration)
    
    def forward(self, task_yaml_path: str):
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

        try:
            input_payload = safe_read_file("curl_command_generated.txt")
            output_payload = safe_read_file("response.json")
            print("Successfully loaded curl_command_generated.txt and response.json")
        except FileNotFoundError as e:
            print(f"Warning: {e}. Using payload from analysis.")
            input_payload = analysis.input_payload
            output_payload = analysis.output_payload

        # Step 2: Generate API integration (frontend -> model server)
        api_integration = self.generate_api_integration(
            task_type=analysis.task_type,
            input_components=analysis.input_components,
            output_components=analysis.output_components,
            input_payload=input_payload[:400],
            output_payload=output_payload[:400],
            model_info=json.dumps(model_info),
            input_type=analysis.input_type,
            output_type=analysis.output_type
        )

        # Step 3: Generate input components
        input_components_html = []
        for comp_type in input_comp_list:
            comp = self.generate_component( 
                task_type=analysis.task_type,
                component_type=comp_type,
                requirements=f"{comp_type} for {analysis.task_type}",
                model_info=json.dumps(model_info),
                input_type=analysis.input_type,
                output_type=analysis.output_type
            )
            input_components_html.append(comp.component_code)
        
        # Step 4: Generate output components
        output_components_html = []
        for comp_type in output_comp_list:
            comp = self.generate_component(
                task_type=analysis.task_type,
                component_type=comp_type,
                requirements=f"{comp_type} for {analysis.task_type}",
                model_info=json.dumps(model_info),
                input_type=analysis.input_type,
                output_type=analysis.output_type
            )
            output_components_html.append(comp.component_code)
        
        # Step 5: Generate complete UI
        complete_ui = self.generate_layout(
            task_name=task_data.get('task_description', {}).get('type', 'ML Task'),
            task_description=task_data.get('task_description', {}).get('description', ''),
            input_components=input_components_html,
            output_components=output_components_html,
            api_integration=api_integration.integration_code
        )
        
        return dspy.Prediction(
            ui_html=complete_ui.complete_html,
            analysis=analysis
        )

class AutoUIGenerator:
    def __init__(self):
        self.ui_generator = UIGenerator()
    
    def generate(self, task_problem_dir: str) -> str:
        task_yaml_path = os.path.join(task_problem_dir, 'task.yaml')
        if not os.path.exists(task_yaml_path):
            raise FileNotFoundError(f"task.yaml not found in {task_problem_dir}")
        
        result = self.ui_generator(task_yaml_path=task_yaml_path)
        return result.ui_html
    
    def save(self, ui_html: str, output_dir: str, task_name: str):
        os.makedirs(output_dir, exist_ok=True)
        
        def clean_code(code):
            return re.sub(r'```(html|javascript)?\n|\n```', '', code).strip()
        
        ui_path = os.path.join(output_dir, f"{task_name}_ui.html")
        with open(ui_path, 'w', encoding='utf-8-sig') as f:
            f.write(clean_code(ui_html))
        
        return ui_path

if __name__ == "__main__":
    generator = AutoUIGenerator()
    task_name = "text_classification"
    task_problem_dir = f"../problems/{task_name}"
    
    try:
        print(f"Generating UI for task: {task_name}")
        print(f"Looking for task.yaml in: {task_problem_dir}")
        
        ui_html = generator.generate(task_problem_dir)
        ui_path = generator.save(ui_html, "../results", task_name)
        
        print("Standalone HTML UI generated successfully!")
        print(f"UI saved to: {ui_path}")

    except Exception as e:
        print(f"Error during generation: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

