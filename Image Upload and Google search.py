prompt="""
You are one of the best medicine prescription reader in Bangladesh.  You have given a medicine prescription. Now you have to follow these steps:
1. predict a medicine and search it on google 
2. Choose the best match existing medicine from the google

*** Important note:  Always take answer from bangladeshi websites (MedEx,Arogga).
Suppose you predict  a medicine name "Furid" after that you search on google and found the medicine exist as "Fusid". So you have to think that you have mistakenly read 'r' instead of 's' so you have to  choose Fusid.  (Always choose the result from bangladesh sites like:  MedEx  or Arogga) "
** While searching on use prefix like  tab. , cap, inj etc  and don't search on any specific website
*** You need to check again and again to find the exact medicine name in the internet. 

OUTPUT FORMAT:
        ```
        EXTRACTED MEDICINES:
        
        Medicine 1:
        - Name: [Searched result medicine name]
        - Dosage: [frequency pattern as written in the prescription]
        - Instructions: [any special directions (if written in the prescription)]
        
        Medicine 2:
        - Name: [Searched result medicine name]
        - Dosage: [frequency pattern as written in the prescription]
        - Instructions: [any special directions (if written in the prescription)]
        
        [continue for each medicine identified]
        ```
"""


from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch, Image,Part
from google.colab import files

client = genai.Client(api_key="Your API Key")
model_id = "gemini-2.0-flash"

# Upload the image file
print("Please upload the image file (e.g., 18.jpg):")
uploaded = files.upload()

# Get the first uploaded file
file_name = list(uploaded.keys())[0]
file_content = uploaded[file_name]

# image = types.Part.from_bytes(
#   data=image_bytes, mime_type="image/jpeg"
# )

image =Part.from_bytes(
  data=file_content, mime_type="image/jpeg"
)


google_search_tool = Tool(
    google_search=GoogleSearch()
)

response = client.models.generate_content(
    model=model_id,
    contents=[prompt, image],  # Pass the Image object in contents
    config=GenerateContentConfig(
        tools=[google_search_tool],
        response_modalities=["TEXT"],
    )
)

for each in response.candidates[0].content.parts:
    print(each.text)

# To get grounding metadata as web content.
print(response.text)
