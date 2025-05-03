from google import genai
from google.genai.types import Part, Tool, GenerateContentConfig, GoogleSearch
from google.colab import files
import concurrent.futures
import time
import nest_asyncio
import re
import difflib
from collections import defaultdict

# Apply nest_asyncio to allow nested event loops (needed for Colab)
nest_asyncio.apply()

# Initialize the client
API_KEY = "---"  # Replace with your actual API key
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

Only focus on identifying medicine names, not other text in the image.
List exactly the number of medicines you see in the prescription - no more, no less.
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
    temperatures = [0.7 + (i * 0.2) for i in range(num_passes)]
    
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
            match = re.search(r'^\d+\.\s+(.*?):\s*(\d+)%', line)
            if match:
                medicine_name = match.group(1).strip()
                confidence = int(match.group(2))
                
                
                # Extract position number
                position_match = re.search(r'^(\d+)\.', line)
                position = int(position_match.group(1)) if position_match else 0
                
                medicine_candidates.append({
                    "name": medicine_name,
                    "confidence": confidence,
                    "position": position
                })
    
    return medicine_candidates

# Function to group similar medicine name candidates
def group_similar_medicines(medicine_candidates):
    # Sort by position first
    medicine_candidates.sort(key=lambda x: x["position"])
    
    # Group by position
    position_groups = defaultdict(list)
    for candidate in medicine_candidates:
        position_groups[candidate["position"]].append(candidate)
    
    # Format the groups for presentation
    formatted_groups = []
    for position in sorted(position_groups.keys()):
        group = position_groups[position]
        
        # Format this group as a list of "Name: Confidence%"
        group_text = f"Medicine {position}: ["
        medicine_texts = []
        for med in group:
            medicine_texts.append(f"{med['name']}: {med['confidence']}%")
        
        group_text += ", ".join(medicine_texts) + "]"
        formatted_groups.append((position, group_text, group))
    
    return formatted_groups

# Function to verify grouped medicines using Google Search
def verify_medicine_groups(medicine_groups):
    verification_prompts = []
    
    for position, group_text, group in medicine_groups:
        # Create a prompt for this group
        prompt = f"""
        I have multiple interpretations of a medicine name from a handwritten prescription (position #{position}):
        
        {group_text}
        
        Please analyze these interpretations and determine the most likely correct medicine name.
        Focus on medicines available in Bangladesh and check pharmaceutical websites like MedEx or Arogga.
        
        Respond with:
        1. The correct medicine name (taken from Medex or Arogga)
        2. The dosage information if available
        3. Brief description of what this medicine is used for
        
        Important: To get better search result carefully choose the medicine name from the group
        """
        verification_prompts.append((position, prompt))
    
    # Process verification in parallel
    verification_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, len(verification_prompts))) as executor:
        future_to_position = {
            executor.submit(
                lambda p: (p[0], client.models.generate_content(
                    model=model_id,
                    contents=p[1],
                    config=GenerateContentConfig(
                        tools=[google_search_tool],
                        response_modalities=["TEXT"],
                        temperature=0.2,
                    )
                ).text),
                prompt
            ): prompt for prompt in verification_prompts
        }
        
        for future in concurrent.futures.as_completed(future_to_position):
            prompt = future_to_position[future]
            try:
                position, result = future.result()
                verification_results.append((position, result))
                print(f"Verified medicine at position {position}")
            except Exception as e:
                print(f"Error verifying medicine at position {prompt[0]}: {e}")
                verification_results.append((prompt[0], f"Error: {str(e)}"))
    
    # Sort by position
    verification_results.sort(key=lambda x: x[0])
    return verification_results

# Function to process final results and format the output
def format_final_results(verification_results):
    final_prompt = """
    You are a medical prescription expert specializing in Bangladeshi medicines. 
    I will provide you with verification results for medicines from a prescription.
    
    For each medicine, create a clean, formatted entry with:
    1. The medicine name (exactly the most correct one, based on the verification)
    2. The dosage information
    3. Any instructions for taking the medicine
    
    Format your response as:
    ```
    FINAL PRESCRIPTION MEDICINES:
    
    1. Medicine Name: 
       Dosage: [dosage info]
       Instructions: [any special instructions]
    
    2. Medicine Name: 
       Dosage: [dosage info]
       Instructions: [any special instructions]
    
    [continue for all medicines in the prescription]
    ```
    
    Here are the verification results for each medicine position:
    
    {VERIFICATION_RESULTS}
    """
    
    # Format the verification results for the prompt
    formatted_results = ""
    for i, (position, result) in enumerate(verification_results, 1):
        formatted_results += f"\n--- Medicine Position {position} ---\n"
        formatted_results += result + "\n"
        formatted_results += "-" * 40 + "\n"
    
    final_prompt = final_prompt.replace("{VERIFICATION_RESULTS}", formatted_results)
    
    # Get final analysis
    response = client.models.generate_content(
        model=model_id,
        contents=final_prompt,
        config=GenerateContentConfig(
            tools=[google_search_tool],
            response_modalities=["TEXT"],
            temperature=0.1,  # Very low temperature for consistent output
        )
    )
    
    return response.text

# Main execution flow
def main():
    print(f"Running multiple interpretation passes in parallel...")
    start_time = time.time()
    
    # Execute all interpretation passes concurrently
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
    
    # Group similar medicine names
    medicine_groups = group_similar_medicines(medicine_candidates)
    print(f"\nGrouped into {len(medicine_groups)} medicine positions")
    
    # Print the groups
    print("\n=== MEDICINE INTERPRETATION GROUPS ===")
    for position, group_text, _ in medicine_groups:
        print(group_text)
    print("-" * 60)
    
    # Verify each medicine group with Google Search
    print("\nVerifying medicine groups with Google Search (this may take some time)...")
    verification_results = verify_medicine_groups(medicine_groups)
    
    verification_time = time.time()
    print(f"Medicine verification completed in {verification_time - interpretation_time:.2f} seconds")

    
    # Format the final results
    print("\nGenerating final analysis with verified medicine names...")
    final_result = format_final_results(verification_results)
    
    end_time = time.time()
    print(f"Final analysis completed in {end_time - verification_time:.2f} seconds")
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    
    # Print final results
    print("\n=== FINAL PRESCRIPTION MEDICINES ===")
    print(final_result)

if __name__ == "__main__":
    main()
