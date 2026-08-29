from dotenv import load_dotenv
from langsmith import traceable
from google import genai
from google.genai import types
from typing import List

load_dotenv()


MAX_AGENT_ITERATIONS = 10
MODEL = "gemini-2.5-flash-lite"
genai_client = genai.Client()


# --- Tools ---
@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up a price of a product in the catalog"""
    print(f"    >> Executing get_product_price({product})")
    prices = {"Asus Tuf F15": 1299.00, "Dell G15": 1199.99}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price
    Available tiers: bronze, sliver, gold."""
    discount_tiers = {"bronze": 0.01, "silver": 0.05, "gold": 0.1}
    return round(price - discount_tiers.get(discount_tier, 0) * price, 2)


tools = [
    {
        "name": "get_product_price",
        "description": "Look up a price of a product in the catalog",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "product": {
                    "type": "string",
                    "description": "The name of the product to look up",
                }
            },
            "required": ["product"],
        },
    },
    {
        "name": "apply_discount",
        "description": (
            "Apply a discount tier to a price and return the final price. "
            "Available tiers: bronze, silver, gold."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "price": {
                    "type": "number",
                    "description": "The original product price",
                },
                "discount_tier": {
                    "type": "string",
                    "description": "The discount tier: bronze, silver, or gold",
                    "enum": ["bronze", "silver", "gold"],
                },
            },
            "required": ["price", "discount_tier"],
        },
    },
]


function_declarations = [types.FunctionDeclaration(**tool) for tool in tools]

tool = types.Tool(function_declarations=function_declarations)


system_instruction = (
    "You are a helpful shopping assistant. "
    "You are a helpful shopping assistant. "
    "You have access to the following tools:\n"
    "1. product price catalog tool\n"
    "2. apply discount tool with discount tiers\n"
    "STRICT RULES: you must follow these exactly:\n"
    "1. Do not ever assume any product price.\n"
    "2. You must call get_product_price first to get the real price.\n"
    "3. Only call apply_discount AFTER you have received "
    "a price from get_product_price. Pass the exact price "
    "returned by get_product_price and don't made up a price.\n"
    "4. Don't calculate discounts yourself, always use apply_discount tool.\n"
    "5. If a user doesn't specify a discount tier ASK them first and do not assume one.\n"
)


@traceable(name="Google AI", run_type="llm")
def google_ai(model: str, contents: List):
    return genai_client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[tool],
        ),
    )


# --- Agent Loop ---
@traceable(name="Google Gen AI SDK Agent Loop")
def run_agent(question: str):

    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    contents = [
        types.UserContent(parts=[types.Part.from_text(text=question)]),
    ]
    for i in range(1, MAX_AGENT_ITERATIONS + 1):
        print(f"[Iteration]: {i}")
        response = google_ai(MODEL, contents)
        print(f"response.funciton_calls: {response.function_calls}")
        if not response.function_calls:
            return response.text
        tool_call = response.function_calls[0]
        print(tool_call)
        tool_name = tool_call.name
        tool_args = tool_call.args
        print(f"[Tools]: tool {tool_name} selected with args {tool_args}")
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found.")
        observation = tool_to_use(**tool_args)

        print(f"[ToolsObservation]: " f"{tool_name} -> {observation}")
        contents.append(response.candidates[0].content)
        contents.append(
            types.Content(
                role="tool",
                parts=[
                    types.Part.from_function_response(
                        name=tool_name, response={"result": observation}
                    )
                ],
            )
        )

    print(f"Error: Max iterations reached without a final answer.")
    return None


if __name__ == "__main__":
    result = run_agent("What is the price of 'Asus Tuf F15' after apply gold discount?")
    print(result)
