import dspy
import yaml
import json
import os
from typing import Dict, List, Any
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
dspy.configure(lm=lm)

class TaskAnalysis(dspy.Signature):
    """Analyze a task.yaml file and extract key information for UI generation."""
    
    task_yaml_content: str = dspy.InputField(desc="Content of the task.yaml file")
    task_type: str = dspy.OutputField(desc="Type of the task (e.g., image_classification)")
    ui_requirements: str = dspy.OutputField(desc="Summary of UI requirements and features needed")
    input_components: list[str] = dspy.OutputField(desc="List of input components needed (e.g., file upload, text input)")
    output_components: list[str] = dspy.OutputField(desc="List of output components needed (e.g., image display, text results)")

class UIComponentGeneration(dspy.Signature):
    """Generate specific UI component code based on task requirements."""
    
    task_type: str = dspy.InputField()
    component_type: str = dspy.InputField(desc="Type of component to generate (input/output/display)")
    requirements: str = dspy.InputField(desc="Specific requirements for this component")
    model_info: str = dspy.InputField(desc="Model API information and format requirements")
    component_code: str = dspy.OutputField(desc="HTML/JavaScript code for the component")

class APIIntegration(dspy.Signature):
    """Generate API integration code for model interaction."""
    
    api_url: str = dspy.InputField()
    input_format: str = dspy.InputField()
    output_format: str = dspy.InputField()
    sample_code: str = dspy.InputField(desc="Sample code or usage notes if available")
    integration_code: str = dspy.OutputField(desc="JavaScript code for API integration")

class UILayoutGeneration(dspy.Signature):
    """Generate complete UI layout combining all components."""
    
    task_description: str = dspy.InputField()
    input_components: list[str] = dspy.InputField()
    output_components: list[str] = dspy.InputField()
    api_integration: str = dspy.InputField()
    complete_html: str = dspy.OutputField(desc="Complete HTML page with CSS and JavaScript")

class UIGenerator(dspy.Module):
    def __init__(self):
        self.analyze_task = dspy.ChainOfThought(TaskAnalysis)
        self.generate_component = dspy.ChainOfThought(UIComponentGeneration)
        self.generate_api_integration = dspy.ChainOfThought(APIIntegration)
        self.generate_layout = dspy.ChainOfThought(UILayoutGeneration)
    
    def forward(self, task_yaml_path: str, data_path: str = None):
        # Load and parse task.yaml
        with open(task_yaml_path, 'r', encoding='utf-8') as f:
            task_yaml_content = f.read()
        
        task_data = yaml.safe_load(task_yaml_content)
        
        # Step 1: Analyze task requirements
        analysis = self.analyze_task(task_yaml_content=task_yaml_content)
        
        # Step 2: Generate API integration code
        model_info = task_data.get('model', {})
        api_integration = self.generate_api_integration(
            api_url=model_info.get('api_url', ''),
            input_format=str(model_info.get('input_format', {})),
            output_format=str(model_info.get('output_format', {})),
            sample_code=model_info.get('sample_code', '')
        )
        
        # Step 3: Generate input components
        input_components = []
        for component_type in analysis.input_components:
            component = self.generate_component(
                task_type=analysis.task_type,
                component_type=f"input_{component_type}",
                requirements=analysis.ui_requirements,
                model_info=str(model_info)
            )
            input_components.append(component.component_code)
        
        # Step 4: Generate output components
        output_components = []
        for component_type in analysis.output_components:
            component = self.generate_component(
                task_type=analysis.task_type,
                component_type=f"output_{component_type}",
                requirements=analysis.ui_requirements,
                model_info=str(model_info)
            )
            output_components.append(component.component_code)
        
        # Step 5: Generate example cases display
        example_component = self.generate_component(
            task_type=analysis.task_type,
            component_type="example_display",
            requirements=f"Display example cases from data path: {data_path}",
            model_info=str(model_info)
        )
        output_components.append(example_component.component_code)
        
        # Step 6: Generate complete UI layout
        complete_ui = self.generate_layout(
            task_description=task_data.get('task', {}).get('description', ''),
            input_components=input_components,
            output_components=output_components,
            api_integration=api_integration.integration_code
        )
        
        return dspy.Prediction(
            task_type=analysis.task_type,
            ui_html=complete_ui.complete_html,
            analysis=analysis,
            api_integration=api_integration.integration_code
        )

class DataProcessor(dspy.Module):
    """Handle data preprocessing and postprocessing"""
    
    def __init__(self):
        self.preprocess = dspy.ChainOfThought("input_data, model_format -> processed_data")
        self.postprocess = dspy.ChainOfThought("model_output, display_format -> formatted_output")
    
    def process_input(self, input_data: Any, model_format: Dict) -> Any:
        """Preprocess input data to match model requirements"""
        # Implementation would depend on specific task type
        pass
    
    def process_output(self, model_output: Any, display_format: Dict) -> Any:
        """Postprocess model output for display"""
        # Implementation would depend on specific task type
        pass

class AutoUIGenerator:
    """Main class that orchestrates the entire UI generation process"""
    
    def __init__(self):
        self.ui_generator = UIGenerator()
        self.data_processor = DataProcessor()
    
    def generate_ui(self, task_problem_dir: str) -> str:
        """
        Generate UI for a given task problem directory
        
        Args:
            task_problem_dir: Path to directory containing task.yaml and data/
            
        Returns:
            HTML string of the generated UI
        """
        task_yaml_path = os.path.join(task_problem_dir, 'task.yaml')
        data_path = os.path.join(task_problem_dir, 'data')
        
        if not os.path.exists(task_yaml_path):
            raise FileNotFoundError(f"task.yaml not found in {task_problem_dir}")
        
        # Generate UI using DSPy
        result = self.ui_generator(task_yaml_path=task_yaml_path, data_path=data_path)
        
        return result.ui_html
    
    def save_ui(self, html_content: str, output_path: str):
        """Save generated UI to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"UI saved to {output_path}")

# Example usage
if __name__ == "__main__":
    # Initialize the generator
    generator = AutoUIGenerator()
    
    # Generate UI for a task problem
    task_problem_dir = "../problems/image_classification"
    ui_html = generator.generate_ui(task_problem_dir)
    generator.save_ui(ui_html, "generated_ui.html")
    
    print("UI Generator System initialized successfully!")

# Additional utility functions for specific task types
class TaskSpecificHandlers:
    """Handlers for specific task types"""
    
    @staticmethod
    def handle_image_classification(task_data: Dict) -> Dict:
        """Specific handling for image classification tasks"""
        return {
            'input_component': 'file_upload_image',
            'output_component': 'classification_results',
            'preprocessing': 'image_resize_normalize',
            'postprocessing': 'confidence_scores_display'
        }
    
    @staticmethod
    def handle_image_to_text(task_data: Dict) -> Dict:
        """Specific handling for image to text tasks"""
        return {
            'input_component': 'file_upload_image',
            'output_component': 'text_display',
            'preprocessing': 'image_encode_base64',
            'postprocessing': 'text_formatting'
        }
    
    @staticmethod
    def handle_text_classification(task_data: Dict) -> Dict:
        """Specific handling for text classification tasks"""
        return {
            'input_component': 'text_input_area',
            'output_component': 'classification_results',
            'preprocessing': 'text_tokenization',
            'postprocessing': 'confidence_scores_display'
        }