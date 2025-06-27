from pipeline import CardiologyLightRAG
import time

def main():
    print("Initializing Cardiology LightRAG system...")
    system = CardiologyLightRAG()
    
    # Default question and context
    query = 'What are the first-line treatments for stable angina?'
    context = 'Patient has diabetes and hypertension'
    
    # Allow user input
    user_query = input(f"Enter your question (or press Enter for default: '{query}'): ")
    if user_query.strip():
        query = user_query
    
    user_context = input(f"Enter patient context (or press Enter for default: '{context}'): ")
    if user_context.strip():
        context = user_context
    
    print("\nProcessing query:", query)
    print("Patient context:", context)
    print("\nPlease wait, this may take some time...")
    
    start_time = time.time()
    answer, explanation = system.process_query(query, context)
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print("CLINICAL ANSWER:")
    print(answer)
    print(f"\nGenerated in {duration:.2f} seconds")
    print("="*80)
    
    print("\nCLINICAL REASONING REPORT:")
    print(explanation)

if __name__ == "__main__":
    main()
