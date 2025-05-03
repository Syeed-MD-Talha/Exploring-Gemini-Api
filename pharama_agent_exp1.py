from google import genai
from google.genai.types import Part, Tool,GenerateContentConfig,GoogleSearch
from google.colab import files
import concurrent.futures
import time
import nest_asyncio
import re

# Apply nest_asyncio to allow nested event loops (needed for Colab)
nest_asyncio.apply()

# Initialize the client
API_KEY = "Your api key"  # Replace with your actual API key
client = genai.Client(api_key=API_KEY)
model_id = "gemini-2.0-flash"

# Configure Google Search Tool
google_search_tool = Tool(
    google_search=GoogleSearch()
)

# Upload the image file
print("Please upload the prescription image:")
uploaded = files.upload()

# Get the first uploaded file
file_name = list(uploaded.keys())[0]
file_content = uploaded[file_name]

image = Part.from_bytes(
    data=file_content, 
    mime_type="image/jpeg"
)

# Create prompt for medicine identification (initial passes)
initial_prompt = """
This image contains a handwritten prescription. Please analyze it and provide:
1. Your interpretation of each medicine name in the prescription
2. For each medicine name, assign a confidence percentage (0-100%)
3. Format each line as "Medicine Name: [confidence]%"
4. Include any dosage information next to each medicine if visible

Only focus on identifying medicine names and dosages, not other text in the image.
Medicine name should be in a list like:
1. --
2. --
"""

# Function to make a single API call for interpretation
def generate_interpretation(temperature):
    response = client.models.generate_content(
        model=model_id,
        contents=[initial_prompt, image],
        config=GenerateContentConfig(temperature=temperature),
    )
    # Run final analysis
    return response.text

# Run all interpretations in parallel using ThreadPoolExecutor
def run_parallel_interpretations(num_passes=5):
    temperatures = [1.0 + (i * 0.1) for i in range(num_passes)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_passes) as executor:
        # Submit all tasks to the executor
        future_to_temp = {executor.submit(generate_interpretation, temp): temp for temp in temperatures}
        
        # Collect results as they complete
        results = []
        for future in concurrent.futures.as_completed(future_to_temp):
            temp = future_to_temp[future]
            try:
                result = future.result()
                print(f"Completed interpretation with temperature {temp:.1f}")
                results.append((temp, result))
            except Exception as e:
                print(f"Error processing temperature {temp}: {e}")
        
    # Sort results by temperature to maintain order
    results.sort()
    return [r[1] for r in results]

# Function to extract a list of possible medicine names from all interpretations
def extract_medicine_candidates(interpretations):
    medicine_candidates = []
    
    for interpretation in interpretations:
        # Look for lines that match the pattern "Medicine Name: XX%" 
        lines = interpretation.strip().split('\n')
        for line in lines:
            # Try to extract medicine name and confidence
            match = re.search(r'^(.*?):\s*(\d+)%', line)
            if match:
                medicine_name = match.group(1).strip()
                confidence = int(match.group(2))
                
                # Check for dosage info after the confidence percentage
                dosage_match = re.search(r'\d+%(.+)$', line)
                dosage = dosage_match.group(1).strip() if dosage_match else ""
                
                medicine_candidates.append({
                    "name": medicine_name,
                    "confidence": confidence,
                    "dosage": dosage
                })
    
    return medicine_candidates

# Function to verify a medicine name using Google Search with focus on Bangladeshi sources
def verify_medicine_with_search(medicine_candidate):
    search_query = f"{medicine_candidate['name']} medicine Bangladesh MedEx Arogga"
    
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=f"""
            I need to verify if a medicine named "{medicine_candidate['name']}" exists in Bangladesh.
            Search for this medicine with focus on Bangladeshi pharmaceutical websites like MedEx or Arogga.
            
            If you find a similar medicine with slightly different spelling, provide that corrected name.
            
            Provide your response in this format:
            - Corrected Name: [verified medicine name]
            - Confidence: [how confident you are this is the correct medicine, on scale 0-100]
            - Source: [website where you found this information]
            - Notes: [any relevant details about the medication or spelling correction]
            """,
            config=GenerateContentConfig(
                tools=[google_search_tool],
                response_modalities=["TEXT"],
            )
        )
        
        return {
            "original": medicine_candidate['name'],
            "verification_result": response.text,
            "dosage": medicine_candidate['dosage']
        }
    except Exception as e:
        print(f"Error verifying medicine {medicine_candidate['name']}: {e}")
        return {
            "original": medicine_candidate['name'],
            "verification_result": f"Error: {str(e)}",
            "dosage": medicine_candidate['dosage']
        }

# Function to extract the final verified medicine information
def process_verification_results(verification_results):
    final_prompt = """
    You are one of the best medicine prescription readers in Bangladesh. Based on multiple verification attempts of medicines from a prescription, provide the final accurate list of medicines.
    
    For each medicine, I'll provide:
    1. The original prediction from the prescription
    2. Verification results from Google searches focusing on Bangladeshi pharmaceutical sources
    3. Dosage information if available
    
    Your task is to determine the most accurate medicine name based on this information. Always prioritize matches from Bangladeshi websites like MedEx or Arogga.
    
    Here are the verification results:
    
    {VERIFICATION_RESULTS}
    
    Provide your final output in this format:
    ```
    EXTRACTED MEDICINES:
    
    Medicine 1:
    - Name: [exactly as written (don't include the power just medicine name only)]
    - Dosage: [frequency pattern]
    - Instructions: [any special directions]
    
    Medicine 2:
    - Name: [exactly as written]
    - Dosage: [frequency pattern]
    - Instructions: [any special directions]
    
    [continue for each medicine identified]
    ```
    
    Important guidelines:
    - List only distinct medicines (remove duplicates)
    - Choose the most accurate name based on verification results
    - Include dosage information when available
    - Extract any special instructions for taking the medicine
    """
    
    # Format the verification results for the prompt
    formatted_results = ""
    for i, result in enumerate(verification_results, 1):
        formatted_results += f"\nMedicine Candidate {i}:\n"
        formatted_results += f"- Original: {result['original']}\n"
        formatted_results += f"- Dosage Info: {result['dosage']}\n"
        formatted_results += f"- Verification:\n{result['verification_result']}\n"
        formatted_results += "-" * 40 + "\n"
    
    final_prompt = final_prompt.replace("{VERIFICATION_RESULTS}", formatted_results)
    
    # Get final analysis with Google search capability
    response = client.models.generate_content(
        model=model_id,
        contents=final_prompt,
        config=GenerateContentConfig(
            tools=[google_search_tool],
            response_modalities=["TEXT"],
            temperature=0.2,  # Lower temperature for more deterministic output
        )
    )
    
    return response.text

# Main execution flow
def main():
    # Execute all interpretation passes concurrently
    print(f"Running 5 interpretation passes in parallel...")
    start_time = time.time()
    
    all_interpretations = run_parallel_interpretations()
    
    interpretation_time = time.time()
    print(f"All interpretation passes completed in {interpretation_time - start_time:.2f} seconds")
    
    # Print all interpretations
    for i, interpretation in enumerate(all_interpretations, 1):
        print(f"\n=== PASS {i} INTERPRETATION ===")
        print(interpretation)
        print("-" * 60)
    
    # Extract medicine candidates from all interpretations
    medicine_candidates = extract_medicine_candidates(all_interpretations)
    print(f"\nExtracted {len(medicine_candidates)} medicine candidates")
    
    # Remove duplicates (case insensitive) while keeping the highest confidence entry
    unique_medicines = {}
    for candidate in medicine_candidates:
        name_lower = candidate['name'].lower()
        if name_lower not in unique_medicines or candidate['confidence'] > unique_medicines[name_lower]['confidence']:
            unique_medicines[name_lower] = candidate
    
    unique_candidates = list(unique_medicines.values())
    print(f"Reduced to {len(unique_candidates)} unique medicine candidates")
    
    # Verify each unique medicine candidate with Google Search
    print("\nVerifying medicines with Google Search (this may take some time)...")
    verification_results = []
    
    # Process verification in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(unique_candidates))) as executor:
        future_to_medicine = {executor.submit(verify_medicine_with_search, med): med for med in unique_candidates}
        
        for future in concurrent.futures.as_completed(future_to_medicine):
            medicine = future_to_medicine[future]
            try:
                result = future.result()
                verification_results.append(result)
                print(f"Verified: {medicine['name']}")
            except Exception as e:
                print(f"Error verifying {medicine['name']}: {e}")
    
    verification_time = time.time()
    print(f"Medicine verification completed in {verification_time - interpretation_time:.2f} seconds")
    
    # Get final analysis
    print("\nGenerating final analysis with verified medicine names...")
    final_result = process_verification_results(verification_results)
    
    end_time = time.time()
    print(f"Final analysis completed in {end_time - verification_time:.2f} seconds")
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    
    # Print final results
    print("\n=== FINAL EXTRACTED MEDICINES ===")
    print(final_result)

if __name__ == "__main__":
    main()
