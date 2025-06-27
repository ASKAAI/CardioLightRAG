# Note: This is a Jupyter Notebook - save with .ipynb extension
import ipywidgets as widgets
from IPython.display import display, Markdown
from pipeline import CardiologyLightRAG
import time

# Initialize system
system = CardiologyLightRAG()

# Create UI components
question_input = widgets.Textarea(
    value='What are the first-line treatments for stable angina?',
    placeholder='Enter a cardiology-related question...',
    description='Question:',
    layout={'width': '90%', 'height': '100px'}
)

patient_context = widgets.Textarea(
    value='Patient has diabetes and hypertension',
    placeholder='Enter patient context (conditions, allergies, meds)...',
    description='Patient:',
    layout={'width': '90%', 'height': '80px'}
)

submit_btn = widgets.Button(description="Get Clinical Answer", button_style='success')
output_area = widgets.Output()
explanation_area = widgets.Output()

def on_submit_clicked(b):
    with output_area:
        output_area.clear_output()
        explanation_area.clear_output()
        
        start_time = time.time()
        print("Processing your cardiology query...")
        
        # Process query
        answer, explanation = system.process_query(
            question_input.value, 
            patient_context.value
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Display results
        display(Markdown(f"### Clinical Answer"))
        display(Markdown(f"{answer}"))
        display(Markdown(f"*Generated in {duration:.2f} seconds*"))
        
        # Show explanation
        with explanation_area:
            display(Markdown(f"### Clinical Reasoning Report"))
            display(Markdown(explanation))

submit_btn.on_click(on_submit_clicked)

# Display UI
display(widgets.VBox([
    widgets.HTML("<h1>Cardiology LightRAG Clinical QA System</h1>"),
    widgets.HTML("<p>Ask cardiology-related questions with patient-specific context</p>"),
    question_input,
    patient_context,
    submit_btn,
    output_area,
    explanation_area
]))