import json
from json import JSONDecodeError
import os
from enum import Enum
from typing import Generic, Type, TypeVar

import groq
from groq import Groq
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel


os.environ["GROQ_API_KEY"] = ""
MODEL = "llama3-70b-8192"

input_text = """

"""

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class ResponseFormat(Enum):
    JSON = "json"
    TEXT = "text"

def predict(prompt: str,
            system_prompt: str = None,
            response_format: ResponseFormat = ResponseFormat.Text,
            model: str = MODEL,
            client: Groq = client
            ):
    messages = [ {"role": "user", "content": prompt, } ]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    try:
        chat_completion = client.chat.completions.create(
            messages = messages,
            model = model,
            temperature = 0.00001,
            response_format = {
                "type": "json_object"
                if response_format == ResponseFormat.JSON
                else "text"
            },
        )
        return chat_completion.choices[0].message.content
    except groq.APIConnectionError as e:
        print("The groq server can not be reached")
        print(e.__cause__)
    except groq.RateLimitError as e:
        print("Too many requests")
        print(e.__cause__)
    except groq.APIStatusError as e:
        print("Not Ok")
        print(e.status_code)
        print(e.message)
        print(e.response)


system_prompt = """
you're evaluating  writing style in text.
your evaluations  must always be in JSON format. Here is an example JSON response:

```
{
    'readability': 4,
    'conciseness': 3
}
```
"""

prompt = f"""
Evaluate the text:

```
{input_text}
```

You're evaluating the readability and conciseness with values from 0 (extremely bad) to 10 (extremely good)
"""

response = predict(prompt, system_prompt, response_format = ResponseFormat.JSON)
print(response)


class WritingScore(BaseModel):
    readability: int
    conciseness: int


schema = {k:v for k,v in WritingScore.schema().items()}
schema = {"properties": schema["properties"], "required": schema["required"]}
print(json.dumps(schema, indent=2))

OUTPUT_FORMAT_INSTRUCTIONS = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type":"array"}}}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema.

Here is the output schema:
```
{schema}
```

Do not return any preamble or explanations, return only a purse JSON string surrounded by triple backticks {```}.
"""

json_instruction = OUTPUT_FORMAT_INSTRUCTIONS.format(
    schema = json.dumps(schema, indent=2)
)
print(json_instruction)

prompt = f"""
Evaluate the writing style of the text:

```
{input_text}
```
Evaluate the readability and conciseness with values from 0 (extremely bad) to 10 (extremely good)

{json_instruction}
"""

print(prompt)

reponse = predict(prompt, MODEL)
print(response)

#From Scratch
response_json = json.load(response.strip("```"))
WritingScore.parse_obj(response_json)

TBaseModel = TypeVar("TBaseModel", bound=BaseModel)

class JsonOutputParser(Generic[TBaseModel]):
    def __init__(self, pydantic_object: Type[TBaseModel]):
        self.pydantic_object = pydantic_object
    
    def parse(self, response: str):
        response_json = json.loads(response.strip("```"))
        return self.pydantic_object.parse_obj(response_json)

parsed_output = JsonOutputParser(pydantic_object = WritingScore).parse(response)
print(parsed_output)

#Using Langchain parser
parser = PydanticOutputParser(pydantic_object = WritingScore)
parsed_output = parser.parse(response)

