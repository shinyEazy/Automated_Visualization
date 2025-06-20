import dspy
import yaml
import json
import os
from typing import Dict, List, Any, Tuple
from pathlib import Path
from dotenv import load_dotenv
import re

load_dotenv()

lm = dspy.LM('openai/gpt-4.1-nano', api_key=os.getenv('OPENAI_API_KEY'))
dspy.configure(lm=lm)

class TaskAnalysis(dspy.Signature):
    """Analyze a task.yaml file and extract key information for UI generation."""
    
    task_yaml_content: str = dspy.InputField(desc="Content of the task.yaml file")
    task_type: str = dspy.OutputField(desc="Type of the task (e.g., image_classification)")
    ui_requirements: str = dspy.OutputField(desc="Summary of UI requirements and features needed")
    input_components: list[str] = dspy.OutputField(desc="List of input components needed (e.g., file upload, text input)")
    output_components: list[str] = dspy.OutputField(desc="List of output components needed (e.g., image display, text results)")
    input_payload: str = dspy.OutputField(desc="Input payload for the model API")
    output_payload: str = dspy.OutputField(desc="Output payload for the model API")
    input_type: str = dspy.OutputField(desc="Type of input (image, text, audio, etc.)")
    output_type: str = dspy.OutputField(desc="Type of output (image, text, json, etc.)")

class UIComponentGeneration(dspy.Signature):
    """Generate specific UI component code based on task requirements with the following design specifications:
    - Use Tailwind CSS classes for styling.
    - Color palette:
      - Light green: rgb(214, 239, 216)
      - Primary green: rgb(128, 175, 129)
      - Secondary green: rgb(80, 141, 78)
      - Dark green: rgb(26, 83, 25)
    - Use monospace font: font-mono
    - Each component must have a unique ID: "{component_type}-{task_type}", e.g., "input-file-image_classification"
    - For containers: p-4 rounded-2xl shadow-lg bg-white
    - For interactive elements: hover:shadow-xl transition duration-300
    - Buttons: bg-[rgb(128,175,129)] text-white px-4 py-2 rounded hover:bg-[rgb(80,141,78)] transition duration-300
    - Inputs: w-full p-2 border border-[rgb(80,141,78)] rounded focus:ring-2 focus:ring-[rgb(128,175,129)] focus:outline-none
    - Specific components:
      - File inputs: <input type="file" class="..."> with accept attribute if specified. For image inputs, use accept="image/*".
      - Text inputs: <textarea> or <input type="text"> based on requirements, with class "w-full p-2 ..."
      - Images: <img class="max-w-full h-auto rounded-lg">, ensure it has an ID for updating src.
      - Text outputs: <p class="text-lg mt-2"> or <div class="text-lg mt-2"> for displaying structured results.
    """
    
    task_type: str = dspy.InputField()
    component_type: str = dspy.InputField(desc="Type of component to generate (e.g., input_file, output_image, output_text)")
    requirements: str = dspy.InputField(desc="Specific requirements for this component")
    model_info: str = dspy.InputField(desc="Model API information and format requirements")
    component_code: str = dspy.OutputField(desc="HTML/JavaScript code for the component with appropriate Tailwind CSS classes")

class APIIntegration(dspy.Signature):
    """
    Generate frontend API integration code with the following requirements:
    - Only handle data collection and submission to backend
    - Convert file inputs to base64 for sending to backend.
    - Send raw input directly to backend endpoint "http://127.0.0.1:5000/predict"
    - Always use POST method.
    - Handle backend response and display results in appropriate output components.
    - Add event listener to submit button (id="submit-btn").
    - Display errors from backend in a dedicated error display area (id="error-message").
    - Support multiple input types (file, text, etc.) by creating a FormData object or JSON payload.
    - For image output, set the src of the output image element.
    - For text/JSON output, update a dedicated div/pre element.
    - Parse the output payload to get the data and display it in the appropriate output component using for each.
    - For text/JSON output:
        - If output is JSON, pretty-print it with JSON.stringify(response, null, 2)
        - If output is array of objects, create formatted display (e.g., table or list)
        - Never display raw [object Object]
    """
    
    input_payload: str = dspy.InputField(desc="Input payload to the server")
    # Emphasize the nested array and the desire for row-by-row display.
    output_payload: str = dspy.InputField(desc="""Output payload from the server.
        Expected structure for text/JSON output: { "data": { "code": "000", "data": [[{ "label": "...", "score": "..." }]], "message": "..." } }.
        For the 'data' array of objects (e.g., [{ "label": "anger", "score": 0.067 }]), display each object's 'label' and 'score' in a separate row within the output component (e.g., a table or list).
    """)
    task_type: str = dspy.InputField(desc="Type of the task")
    input_components: list[str] = dspy.InputField(desc="List of input component types (e.g., file, text)")
    output_components: list[str] = dspy.InputField(desc="List of output component types (e.g., image, text)")
    integration_code: str = dspy.OutputField(desc="JavaScript code for API integration")

class UILayoutGeneration(dspy.Signature):
    """Generate complete UI layout combining all components with the following design specifications:
    - Use Tailwind CSS via CDN: <script src="https://cdn.tailwindcss.com"></script>
    - Overall page:
      - Background: bg-[rgb(214,239,216)]
      - Text color: text-[rgb(26,83,25)]
      - Font: font-mono
    - Structure:
      - Wrap in <div class="min-h-screen bg-[rgb(214,239,216)] text-[rgb(26,83,25)] font-mono">
      - Header: <div class="bg-[rgb(128,175,129)] text-white p-4 sticky top-0 z-10"> with <h1 class="text-2xl font-bold">{task_name}</h1></p>
      - Main content: <main class="container mx-auto p-4">
      - Layout: <div class="flex flex-col md:flex-row gap-6">
        - Inputs: <div class="flex-1 bg-white p-6 rounded-2xl shadow-lg hover:shadow-xl transition duration-300"> with input_components and <button id="submit-btn" class="mt-4 bg-[rgb(128,175,129)] text-white px-4 py-2 rounded hover:bg-[rgb(80,141,78)] transition duration-300">Submit</button> and <div id="error-message" class="text-red-500 mt-2"></div>
        - Outputs: <div class="flex-1 bg-white p-6 rounded-2xl shadow-lg hover:shadow-xl transition duration-300"> with output_components
    - Include api_integration in <script> at the end of <body>
    - Ensure responsiveness: Stack vertically on small screens, side-by-side on medium+ screens
    """
    
    task_name: str = dspy.InputField(desc="Name of the task")
    task_description: str = dspy.InputField(desc="Description of the task")
    input_components: list[str] = dspy.InputField(desc="List of HTML strings for input components")
    output_components: list[str] = dspy.InputField(desc="List of HTML strings for output components")
    api_integration: str = dspy.InputField(desc="JavaScript code for API integration")
    complete_html: str = dspy.OutputField(desc="Complete HTML page with CSS and JavaScript, styled according to the design specifications")

class BackendGeneration(dspy.Signature):
    """Generate Python Flask backend code with the following specifications:
    
    1. Framework & Setup:
    - Use Flask with flask_cors.CORS
    - Create an endpoint ('/predict')
    
    2. Prediction Endpoint (POST /predict):
    a) Input Handling:
       - Accept JSON payload with structure: {{input_payload}}
       - Validate input_type and data presence
    
    b) Processing Pipeline:
       IMAGE PATH:
        1. Decode base64 to bytes
        2. Send to model API: {{input_payload}}
        3. Process response:
           - Parse JSON, extract predictions: result['data']
           - Load original image with PIL
           - Draw bounding boxes + labels using prediction data
           - Encode processed image to base64
        4. Return: {{output_payload}}
        
       TEXT PATH:
        1. Directly send text payload to model API: {{input_payload}}
        2. Return: {{output_payload}}
    
    c) Model API Call:
       - URL: from `model_info.api_url`
       - Use requests.post()
    
    3. Response Standardization:
    - Always return JSON: {{output_payload['data']}}
    
    Note: Maintain clear separation between image/text pipelines.
    Use helper functions for modularity.
    """
    
    input_payload: str = dspy.InputField(desc="Input payload for the model API")
    output_payload: str = dspy.InputField(desc="Output payload for the model API")
    task_type: str = dspy.InputField(desc="Type of the task (e.g., object_detection)")
    model_info: str = dspy.InputField(desc="Model API information and format requirements, including api_url, input_format, and output_format.")
    backend_code: str = dspy.OutputField(desc="Python Flask backend code")

class UIGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyze_task = dspy.ChainOfThought(TaskAnalysis)
        self.generate_component = dspy.ChainOfThought(UIComponentGeneration)
        self.generate_api_integration = dspy.ChainOfThought(APIIntegration)
        self.generate_layout = dspy.ChainOfThought(UILayoutGeneration)
        self.generate_backend = dspy.ChainOfThought(BackendGeneration)
    
    def forward(self, task_yaml_path: str, data_path: str = None):
        # Load and parse task.yaml
        with open(task_yaml_path, 'r', encoding='utf-8') as f:
            task_yaml_content = f.read()
        
        task_data = yaml.safe_load(task_yaml_content)
        model_info = task_data.get('model_information', {})

        # Step 1: Analyze task requirements
        analysis = self.analyze_task(task_yaml_content=task_yaml_content)

        # Step 2: Generate backend code
        backend = self.generate_backend(
            task_type=analysis.task_type,
            input_payload=analysis.input_payload,
            output_payload=analysis.output_payload,
            model_info=json.dumps(model_info)
        )

        # Step 3: Generate API integration for frontend
        COMPONENT_MAPPING = {
            "image": {
                "input": "input_file",
                "output": "output_image"
            },
            "text": {
                "input": "input_text",
                "output": "output_text"
            },
            "json": {
                "output": "output_json"
            }
        }

        input_component_type = COMPONENT_MAPPING[analysis.input_type]["input"]
        output_component_type = COMPONENT_MAPPING[analysis.output_type]["output"]

        api_integration = self.generate_api_integration(
            task_type=analysis.task_type,
            input_components=input_component_type,
            output_components=output_component_type,
            input_payload="{\"texts\": List of strings} or {\"images\": List of images}",
            output_payload="""
{
    "data": {
        "code": "000",
        "data": [
            [
                {
                    "label": "anger",
                    "score": 0.10536875575780869
                },
                {
                    "label": "disgust",
                    "score": 0.15130650997161865
                },
                {
                    "label": "fear",
                    "score": 0.030560608953237534
                },
                {
                    "label": "joy",
                    "score": 0.01893160678446293
                },
                {
                    "label": "neutral",
                    "score": 0.6386061906814575
                },
                {
                    "label": "sadness",
                    "score": 0.03592739254236221
                },
                {
                    "label": "surprise",
                    "score": 0.019298972561955452
                }
            ]
        ],
        "message": "Thành công"
    }
}
"""
        )

        # Step 4: Generate input components
        input_components_html = []
        # Based on task_yaml: input is a single image
        input_components_html.append(self.generate_component(
            task_type=analysis.task_type,
            component_type="input_file",
            requirements="Allow user to upload a single image file for object detection.",
            model_info=json.dumps(model_info)
        ).component_code)
        
        # Step 5: Generate output components
        output_components_html = []
        # Based on task_yaml: output is the image with bounding boxes and detection details
        output_components_html.append(self.generate_component(
            task_type=analysis.task_type,
            component_type="output_image",
            requirements="Display the input image with predicted bounding boxes drawn on it.",
            model_info=json.dumps(model_info)
        ).component_code)
        output_components_html.append(self.generate_component(
            task_type=analysis.task_type,
            component_type="output_text",
            requirements="Display the detected objects' class names, confidence scores, and bounding box coordinates.",
            model_info=json.dumps(model_info)
        ).component_code)
        
        # Step 7: Generate complete UI layout
        complete_ui = self.generate_layout(
            task_name=task_data.get('task_description', {}).get('type', 'Untitled Task'),
            task_description=task_data.get('task_description', {}).get('description', 'No description provided.'),
            input_components=input_components_html,
            output_components=output_components_html,
            api_integration=api_integration.integration_code
        )
        
        return dspy.Prediction(
            task_type=analysis.task_type,
            ui_html=complete_ui.complete_html,
            backend_code=backend.backend_code,
            analysis=analysis
        )

class AutoUIGenerator:
    def __init__(self):
        self.ui_generator = UIGenerator()
    
    def generate(self, task_problem_dir: str) -> Tuple[str, str]:
        """
        Generate UI and backend for a given task problem directory
        
        Args:
            task_problem_dir: Path to directory containing task.yaml and data/
            
        Returns:
            Tuple (html_content, backend_code)
        """
        task_yaml_path = os.path.join(task_problem_dir, 'task.yaml')
        data_path = os.path.join(task_problem_dir, 'data')
        
        if not os.path.exists(task_yaml_path):
            raise FileNotFoundError(f"task.yaml not found in {task_problem_dir}")
        
        # Generate UI and backend using DSPy
        result = self.ui_generator(task_yaml_path=task_yaml_path, data_path=data_path)
        return result.ui_html, result.backend_code
    
    def save(self, ui_html: str, backend_code: str, output_dir: str, task_name: str):
        """Save generated UI and backend to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Clean code outputs (remove markdown code fences)
        def clean_code(code):
            # Remove triple backticks and optional language specifier, and ensure no empty lines from it
            cleaned = re.sub(r'```(html|python)?\n|\n```', '', code).strip()
            return cleaned
        
        # Save frontend
        ui_path = os.path.join(output_dir, f"{task_name}_ui.html")
        with open(ui_path, 'w', encoding='utf-8') as f:
            f.write(clean_code(ui_html))
        
        # Save backend
        backend_path = os.path.join(output_dir, f"{task_name}_backend.py")
        with open(backend_path, 'w', encoding='utf-8') as f:
            f.write(clean_code(backend_code))
        
        print(f"Frontend saved to {ui_path}")
        print(f"Backend saved to {backend_path}")

if __name__ == "__main__":
    generator = AutoUIGenerator()
    task_name = "object_detection_in_image"
    task_problem_dir = f"../problems/{task_name}"
    
    try:
        ui_html, backend_code = generator.generate(task_problem_dir)
        generator.save(ui_html, backend_code, "../results", task_name)
        print("UI and Backend generated successfully!")
        
        # Print backend usage instructions
        print("\nHOW TO USE THE BACKEND:")
        print("1. Install requirements: pip install flask requests python-dotenv")
        print("2. Create a .env file with your API keys")
        print(f"3. Run the backend: python results/{task_name}_backend.py")
        print(f"4. Open the UI in browser: results/{task_name}_ui.html")
        
    except Exception as e:
        print(f"Error during generation: {str(e)}")