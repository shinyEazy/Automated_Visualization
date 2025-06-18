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
      - Example display: List examples in <div class="p-4 bg-gray-100 rounded-lg mb-4"> with a button <button class="bg-[rgb(80,141,78)] text-white px-3 py-1 rounded hover:bg-[rgb(26,83,25)]" onclick="loadExample(index)">Load Example</button>, include <script> with const examples = [...]
    """
    
    task_type: str = dspy.InputField()
    component_type: str = dspy.InputField(desc="Type of component to generate (e.g., input_file, output_image, output_text)")
    requirements: str = dspy.InputField(desc="Specific requirements for this component")
    model_info: str = dspy.InputField(desc="Model API information and format requirements")
    component_code: str = dspy.OutputField(desc="HTML/JavaScript code for the component with appropriate Tailwind CSS classes")

class APIIntegration(dspy.Signature):
    """Generate frontend API integration code with the following requirements:
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
    """
    
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
      - Header: <div class="bg-[rgb(128,175,129)] text-white p-4 sticky top-0 z-10"> with <h1 class="text-2xl font-bold">{task_name}</h1> and <p class="mt-1">{task_description}</p>
      - Main content: <main class="container mx-auto p-4">
      - Layout: <div class="flex flex-col md:flex-row gap-6">
        - Inputs: <div class="flex-1 bg-white p-6 rounded-2xl shadow-lg hover:shadow-xl transition duration-300"> with input_components and <button id="submit-btn" class="mt-4 bg-[rgb(128,175,129)] text-white px-4 py-2 rounded hover:bg-[rgb(80,141,78)] transition duration-300">Submit</button> and <div id="error-message" class="text-red-500 mt-2"></div>
        - Outputs: <div class="flex-1 bg-white p-6 rounded-2xl shadow-lg hover:shadow-xl transition duration-300"> with output_components
      - Examples (if provided): <div class="mt-6 bg-white p-6 rounded-2xl shadow-lg">{example_component}</div>
    - Include api_integration in <script> at the end of <body>
    - Ensure responsiveness: Stack vertically on small screens, side-by-side on medium+ screens
    """
    
    task_name: str = dspy.InputField(desc="Name of the task")
    task_description: str = dspy.InputField(desc="Description of the task")
    input_components: list[str] = dspy.InputField(desc="List of HTML strings for input components")
    output_components: list[str] = dspy.InputField(desc="List of HTML strings for output components")
    example_component: str = dspy.InputField(desc="HTML string for example display component", default=None)
    api_integration: str = dspy.InputField(desc="JavaScript code for API integration")
    complete_html: str = dspy.OutputField(desc="Complete HTML page with CSS and JavaScript, styled according to the design specifications")

class BackendGeneration(dspy.Signature):
    """Generate Python Flask backend code for processing user inputs, calling an external model API, and processing its output.
    
    Requirements:
    - Use Flask framework with CORS.
    - Create a root endpoint: `@app.route('/')` which can return a simple message.
    - Create a prediction endpoint: `@app.route('/predict', methods=['POST'])`.
    - Handle image file uploads:
        - Receive base64 encoded image data from the frontend.
        - Decode the base64 string to bytes.
    - Call the external model API as specified in model_info.api_url.
        - The external API expects a JSON payload: `{"data": "base64_encoded_image_string"}`.
        - Use `requests` library for the API call.
    - Process the model API response:
        - The model API returns a list of objects with 'class', 'class_name', 'confidence', 'bbox'.
        - Use Pillow (PIL) to load the *original* input image (decoded from base64).
        - Draw bounding boxes and class names onto this image based on the model's predictions.
        - Re-encode the *processed image* (with bounding boxes) back to base64.
    - Return a JSON response to the frontend: `{"success": bool, "result": {"processed_image": "base64_encoded_image", "detections": [...]}, "error": optional_message}`.
    - Include comprehensive error handling for file processing, API calls, and image drawing.
    """
    
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
            model_info=json.dumps(model_info) # Pass model_info as a JSON string
        )

        # Step 3: Generate API integration for frontend
        # Extract component types from analysis.input_components and analysis.output_components
        # For this specific object detection task, input is 'image' (file upload), output is 'image' (display) and 'text' (detections)
        frontend_input_types = ['file'] # Assuming 'file' for image upload
        frontend_output_types = ['image', 'text'] # Image with boxes, and detection details

        api_integration = self.generate_api_integration(
            task_type=analysis.task_type,
            input_components=frontend_input_types,
            output_components=frontend_output_types
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
        
        # Step 6: Generate example cases display (simplified as data_path might not be directly usable for in-browser examples without server-side support)
        example_component = None
        # For object detection, handling examples directly in frontend without a server to serve images is complex.
        # This part might need further refinement for a robust solution.
        # For now, we'll generate a placeholder or skip if `data_path` is present but not configured for client-side loading.
        if data_path and os.path.exists(data_path):
            # A more sophisticated approach would be needed here to read data and prepare it for JS
            # For now, we'll just indicate that examples are available.
            example_component = f"""
            <div class="p-4 bg-gray-100 rounded-lg mb-4">
                <h3 class="text-xl font-semibold mb-2">Example Cases</h3>
                <p>Example data is available in the '{data_path}' directory. Loading examples directly in the browser requires server-side setup.</p>
                <button class="bg-[rgb(80,141,78)] text-white px-3 py-1 rounded hover:bg-[rgb(26,83,25)]" onclick="alert('Loading examples not yet implemented for direct client-side display without server support.')">Load Example</button>
            </div>
            """
        
        # Step 7: Generate complete UI layout
        complete_ui = self.generate_layout(
            task_name=task_data.get('task_description', {}).get('type', 'Untitled Task'),
            task_description=task_data.get('task_description', {}).get('description', 'No description provided.'),
            input_components=input_components_html,
            output_components=output_components_html,
            example_component=example_component,
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

# Example usage
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